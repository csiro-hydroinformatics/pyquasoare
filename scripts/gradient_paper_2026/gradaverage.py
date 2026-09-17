#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-07-09 15:48:24.894963
## Comment : Benchmark gradient estimation against analytical solution and numerical difference
##
## ------------------------------

import sys
import time
import re
import math
import argparse
from pathlib import Path
from collections import namedtuple

import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.stat import armodels

from pyquasoare import gradients

import gradbench

import importlib
importlib.reload(gradients)
importlib.reload(gradbench)

from utils import get_parameters

from gradbench import get_data, get_reservoir_id, \
    configure_model, allocate_model, log_model, set_qstars, \
    set_parameters, data_reader, REPORT_PROBS

np.random.seed(5446)


def get_options():
    timestep = "daily"
    # Duration depends on number of reaches
    duration_base = {1: 100, 2: 100, 3: 50, 5: 20, 20: 5, 50: 5}
    duration_repeat = 1

    nuh = 5
    nsubsteps = [1, 6, 12, 24, 144]

    stationids = data_reader.SITEIDS

    nreaches_ref = 5
    nreaches = [1, nreaches_ref]

    nalphas_ref = 100
    nalphas = [nalphas_ref]

    nrepeat = 20
    nbatch = 1
    batches = np.arange(nbatch)

    _, X1, _, X3, _ = get_parameters()
    gamma_means = [
        (X1, X3, X1),
        (X1 * 10, X3 * 10, X1 * 10),
        (X1 / 10, X3 / 10, X1 / 10)
        ]
    gamma_std = 0.2

    demand_mean = -0.5
    demand_std = 0.2

    x4_mean = 1.0
    x4_std = 0.2

    opm = hyruns.OptionManager(duration_base=duration_base,
                               duration_repeat=duration_repeat,
                               gamma_std=gamma_std,
                               timestep=timestep,
                               demand_mean=demand_mean,
                               demand_std=demand_std,
                               x4_mean=x4_mean,
                               x4_std=x4_std,
                               nrepeat=nrepeat,
                               nuh=nuh,
                               nsubsteps=nsubsteps,
                               nalphas_ref=nalphas_ref,
                               nreaches_ref=nreaches_ref,
                               nbatch=nbatch)

    opm.from_cartesian_product(batch=batches,
                               stationid=stationids,
                               nreaches=nreaches,
                               nalphas=nalphas,
                               gamma_mean=gamma_means)
    return opm


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "data"
    fout = froot / "outputs" / f"{source_file.stem}_v{config.version}"
    flogs = froot / "logs" / source_file.stem

    if config.debug:
        fout = flogs / fout.stem

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fout", "flogs"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fout, flogs)
    flogs.mkdir(exist_ok=True, parents=True)
    fout.mkdir(exist_ok=True, parents=True)

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    fl = script_paths.flogs / f"{basename}_TASK{config.taskid}.log"
    logger = iutils.get_logger(basename, flog=fl, console=True)
    dd = config._asdict()
    dd.pop("options")
    dd.pop("task")
    logger.log_dict(dd, "Configuration")
    logger.info("", nret=1)
    logger.info(f"Number of tasks: {config.options.ntasks}")

    return logger


def process(config, script_paths, logger, data):
    # Prepare output folders
    fperf = script_paths.fout / "performance"
    fperf.mkdir(exist_ok=True)

    fsims = script_paths.fout / "simulations"
    fsims.mkdir(exist_ok=True)

    # Setup up model
    mods = {}
    hclim = data.hydroclim
    if config.debug:
        hclim = hclim.loc["2022-01":"2022-12"]
    istart, iend = 0, len(hclim) - 1
    time = hclim.index[istart:iend]
    hclim = hclim.values[istart:iend]

    nsubsteps = config.options.context["nsubsteps"]
    nsubsteps.sort()
    assert nsubsteps[0] == 1

    for nsub in nsubsteps:
        config = config._replace(nsubsteps=nsub)
        mod = configure_model(config, logger, data.area)
        allocate_model(config, logger, mod, hclim)
        mods[nsub] = mod

    # Define job
    batch = config.task.batch
    nbatch = config.options.context["nbatch"]
    nrepeat = 1 if config.debug else config.options.context["nrepeat"]
    ibatches = np.array_split(np.arange(nrepeat), nbatch)[batch]

    # initialise perf results
    t1 = np.random.uniform(0, 1, (10, 2, 2))
    ex, _ = gradients.compare(t1, t1)
    mnames = [f"nsubsteps{nsub:03d}" for nsub in nsubsteps[:-1]]
    cc = [f"fx_log10errmax{p:02d}_{m}[-]" for m in mnames
          for p in REPORT_PROBS]
    cc += [f"dfx_log10errmax{p:02d}_{m}[-]" for m in mnames
           for p in REPORT_PROBS]
    results = pd.DataFrame(np.nan, index=ibatches, columns=cc)

    mod = mods[1]
    results.loc[:, "ntimesteps_agg"] = mod.ntimesteps_agg
    results.loc[:, "nreservoirs"] = mod.nreservoirs
    results.loc[:, "nvariables"] = len(mod.variables_reservoir_id)

    # repeat experiment
    logger.info("")
    for irepeat in ibatches:
        logger.info(f"## Repeat #{irepeat + 1} / {len(ibatches)} ##",
                    nret=2)

        # Set random model inputs and run models
        for nsub in nsubsteps:
            mod = mods[nsub]
            logger.info(f"Run Quasoare - nsubsteps={nsub}", nret=1)
            if nsub == 1:
                config = config._replace(nsubsteps=1)
                set_qstars(config, logger, mod, hclim)
                set_parameters(config, logger, mod)
            else:
                mod1 = mods[1]
                mod.set_Qstars(mod1.Qstars)
                mod.gammas = mod1.gammas
                for res in np.arange(mod.nreservoirs):
                    mod.coefs_linear[res] = mod1.coefs_linear[res]

            if irepeat == 0 and nsub == 1:
                log_model(config, logger, mod)

            mod.run()
            mod.aggregate()

            # Rescale fluxes -> Get dfluxes in mm / d
            nvars = len(mod.variables_reservoir_id)
            for ivar in range(nvars):
                rid = mod.variables_reservoir_id[ivar]
                vtid = mod.variables_type_id[ivar]
                if vtid == 0:
                    gam = mod.gammas[rid]
                    mod.dfluxes[:, ivar] *= mod.gammas[rid]
                    mod.dfluxes_agg[:, ivar] *= mod.gammas[rid]

        # Compute performance
        nsubmax = np.max(nsubsteps)
        modsmax = mods[nsubmax]
        fxa = modsmax.fluxes_agg
        dfxa = modsmax.dfluxes_agg

        for nsub, mod in mods.items():
            if nsub == nsubmax:
                continue

            nagg = nsubmax // nsub
            assert nagg * nsub == nsubmax
            ex, st = gradients.compare(fxa[:, None, :], mod.fluxes_agg[:, None, :])

            for prob, val in ex.items():
                cn = f"fx_log10errmax{prob:02d}_nsubsteps{nsub:03d}[-]"
                le = math.log10(val) if val > 0 else np.nan
                results.loc[irepeat, cn] = le

            ex, st = gradients.compare(dfxa, mod.dfluxes_agg)
            for prob, val in ex.items():
                cn = f"dfx_log10errmax{prob:02d}_nsubsteps{nsub:03d}[-]"
                le = math.log10(val) if val > 0 else np.nan
                results.loc[irepeat, cn] = le

            #if config.debug and ex.max() > 1e-1:
            #    imax = np.where(st[90] > 1e-1)
            #    ivar, res, ifx = [ii[0] for ii in imax]
            #    rid = mod.variables_reservoir_id[ivar]
            #    tid = mod.variables_type_id[ivar]
            #    emax = st[90][ivar, res, ifx]

            #    import matplotlib.pyplot as plt
            #    plt.close("all")
            #    fig, axs = plt.subplots(nrows=3, figsize=(12, 9),
            #                            sharex=True,
            #                            layout="constrained")
            #    ax = axs[0]
            #    k = modsmax.idx_end_timestep_agg
            #    y1 = modsmax.x[k, res]
            #    ax.plot(time, y1, label="submax")

            #    k = mod.idx_end_timestep_agg
            #    y2 = mod.x[k, res]
            #    ax.plot(time, y2, label=f"sub{nsub}")
            #    ax.legend()

            #    ax = axs[1]
            #    #y0 = -fxa[:, res - 1, 5]
            #    #ax.plot(time, y0, "k--", label="inflow", lw=0.5)
            #    #yby = -fxa[:, res, 3]
            #    #ax.plot(time, yby, "k:", label="bypass", lw=0.5)

            #    y1 = fxa[:, res, ifx]
            #    y2 = mod.fluxes_agg[:, res, ifx]
            #    ax.plot(time, y1, label=f"sub{modsmax.nsubsteps}")
            #    ax.plot(time, y2, label=f"sub{nsub}")
            #    ax.legend()

            #    ax = axs[2]
            #    y1 = dfxa[:, ivar, res, ifx]
            #    y2 = mod.dfluxes_agg[:, ivar, res, ifx]
            #    ax.plot(time, y1, label=f"sub{modsmax.nsubsteps}")
            #    ax.plot(time, y2, label=f"sub{nsub}")

            #    tax = ax.twinx()
            #    tax.plot(time, y2/y1, color="0.5", lw=0.5)

            #    ftitle = f"Res {res} - Var {ivar} ({rid}/{tid}) - Fx {ifx} - Em {emax:0.2e}"
            #    fig.suptitle(ftitle, fontweight="bold")

            #    plt.show()
            #    import pdb; pdb.set_trace()

    results.index.name = "REPEAT"
    fres =  fperf / f"performance_v{version}_TASK{taskid}.csv"
    csv.write_csv(results, fres, "Performance",
                  script_paths.source_file,
                  write_index=True,
                  lineterminator="\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark gradient "
                                                 + "estimation against numerical difference",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version",
                        help="Version number",
                        type=int, required=True)
    parser.add_argument("-t", "--taskid", help="JobID",
                        type=int, default=-1)
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    parser.add_argument("-o", "--overwrite", help="Overwrite data",
                        action="store_true", default=False)
    args = parser.parse_args()

    # Config
    version = args.version
    overwrite = args.overwrite
    debug = args.debug
    taskid = 0 if debug else args.taskid
    nsubsteps = 0

    Config = namedtuple("Config",
                        ["version", "taskid", "overwrite",
                         "debug", "options", "task", "nsubsteps"])
    opm = get_options()
    task = opm.get_task(taskid)

    if debug:
        task.nreaches = 5
        _, X1, _, X3, _ = get_parameters()
        task.gamma_mean = (X1, X3, X1)

    config = Config(version, taskid, overwrite,
                    debug, opm, task, nsubsteps)

    # Baseline
    script_paths = get_script_paths(config)

    fopm = script_paths.fout / "option_manager.json"
    opm.save(fopm)

    logger = get_logger(config, script_paths)
    task.log(logger)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
