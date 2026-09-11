#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-07-09 15:48:24.894963
## Comment : Benchmark gradient estimation against analytical solution and numerical difference
##
## ------------------------------


import sys
import os
import time
import re
import json
import math
import argparse
from pathlib import Path
from collections import namedtuple

import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import dates as mdates

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.stat import armodels

from pyquasoare import gradients

from utils import get_parameters

import importlib
import utils
importlib.reload(utils)

from gradbench import get_data, get_reservoir_id, \
    configure_model, allocate_model, log_model, set_qstars, \
    set_parameters, comp_dfluxes, data_reader, \
    MODEL_TIMESTEP

from utils import comp_dfluxes, REPORT_PROBS

np.random.seed(5446)


def aggregate(x, ntimesteps, nagg):
    agg = x.cumsum(axis=0)
    agg[nagg:] = agg[nagg:] - agg[:-nagg]
    idx = np.arange(nagg - 1, (ntimesteps - 1) * nagg, nagg)
    return agg[idx] / nagg


def get_options():
    timestep = "daily"
    # Duration depends on number of reaches
    duration_base = {1: 100, 2: 100, 3: 50, 5: 20, 20: 5, 50: 5}
    duration_repeat = 1

    nuh = 5
    nsubsteps = [1, 12]

    stationids = data_reader.SITEIDS

    nreaches_ref = 5
    nreaches = [1, nreaches_ref, 50]

    nalphas_ref = 100
    nalphas = [nalphas_ref]

    nrepeat = 20
    nbatch = 1
    batches = np.arange(nbatch)

    X0, X1, X2, X3, X4 = get_parameters()
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
    #hclim = hclim.loc["2022"].values
    #istart, iend = 0, len(hclim) - 1
    istart, iend = 0, 70
    time = hclim.index[istart:iend]
    hclim = hclim.values[istart:iend]

    nsubsteps = config.options.context["nsubsteps"]
    nsubsteps.sort()

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
    mnames = [f"nsubsteps{nsub:03d}" for nsub in nsubsteps[:-1]]
    cc = [f"fx_log10errmax{p:02d}_{m}[-]" for m in mnames
          for p in REPORT_PROBS]
    cc += [f"dfx_log10errmax{p:02d}_{m}[-]" for m in mnames
          for p in REPORT_PROBS]
    results = pd.DataFrame(np.nan, index=ibatches, columns=cc)

    mod = mods[1]
    results.loc[:, "ntimesteps"] = mod.ntimesteps
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
                #set_qstars(config, logger, mod, hclim, False)
                set_parameters(config, logger, mod)
            else:
                mod1 = mods[1]
                mod.Qstars = np.repeat(mod1.Qstars, nsub, axis=0)
                mod.gammas = mod1.gammas
                for res in np.arange(mod.nreservoirs):
                    mod.coefs_linear[res] = mod1.coefs_linear[res]

            if irepeat == 0:
                log_model(config, logger, mod)

            mod.run()

            # Rescale fluxes -> Get dfluxes in mm / d
            nvars = len(mod.variables_reservoir_id)
            for ivar in range(nvars):
                res = mod.variables_reservoir_id[ivar]
                mod.dfluxes[:, ivar] *= mod.gammas[res]

        # Compute performance
        nsubmax = np.max(nsubsteps)
        modsmax = mods[nsubmax]

        for nsub, mod in mods.items():
            if nsub == nsubmax:
                continue

            if nsub != 1:
                continue

            nagg = nsubmax // nsub
            assert nagg * nsub == nsubmax
            fx = aggregate(modsmax.fluxes, mod.ntimesteps, nagg)
            ex = comp_dfluxes(fx, fx[:, None, :], mod.fluxes[:-1, None, :])

            #import matplotlib.pyplot as plt
            #plt.close("all")
            #fig, axs = plt.subplots(nrows=3, ncols=5, sharex=True,
            #                        layout="constrained",
            #                        figsize=(15, 7))

            #n = modsmax.nsubsteps
            #delta = np.array([pd.Timedelta(days=i/n) for i in range(1, n+1)])
            #delta = np.tile(delta,(len(time)))
            #t0 = np.repeat(time, n) + delta

            #n = mod.nsubsteps
            #delta = np.array([pd.Timedelta(days=i/n) for i in range(1, n+1)])
            #delta = np.tile(delta,(len(time)))
            #t1 = np.repeat(time, n) + delta

            #res0 = get_reservoir_id(config, 0, "openwater")
            #res1 = get_reservoir_id(config, 0, "gr4jprod")
            #res2 = get_reservoir_id(config, 0, "uh0")
            #res3 = get_reservoir_id(config, 0, "uh1")
            #res4 = get_reservoir_id(config, 0, "gr4jroute")

            ##pstart = mdates.date2num(pd.to_datetime("2019-01-01"))
            ##pend = mdates.date2num(pd.to_datetime("2022-12-31"))
            #pstart = mdates.date2num(time[0])
            #pend = mdates.date2num(time[-1])

            #for ires, res in enumerate([res0, res1, res2, res3, res4]):
            #    ax = axs[0, ires]
            #    ax.plot(t0, modsmax.Qstars[:, res, :], "+-")
            #    ax.plot(t1, mod.Qstars[:, res, :], "o-")

            #    ax = axs[2, ires]
            #    ax.plot(t0, -modsmax.fluxes[:, res, 5], "+-")
            #    ax.plot(t1[:-1], -fx[:, res, 5], "o-")
            #    ax.plot(t1, -mod.fluxes[:, res, 5], "o-")
            #    tax = ax.twinx()
            #    tax.plot(t1[:-1], np.abs(fx[:, res, 5]-mod.fluxes[:-1, res, 5]),
            #             color="0.1", lw=0.8, alpha=0.1)

            #    ax.set(title=f"Res {res} - fluxes",
            #           xlim=(pstart, pend))

            #    ax = axs[1, ires]
            #    ax.plot(t0, modsmax.x[:, res], "+-")
            #    ax.plot(t1, mod.x[:, res], "o-")
            #    ax.set(title=f"Res {res} - stores",
            #           xlim=(pstart, pend))

            #plt.show()
            #import pdb; pdb.set_trace()

            for prob in REPORT_PROBS:
                cn = f"fx_log10errmax{prob:02d}_nsubsteps{nsub:03d}[-]"
                le = math.log10(ex[prob]) if ex[prob] > 0 else np.nan
                results.loc[irepeat, cn] = le

            dfx = aggregate(modsmax.dfluxes, mod.ntimesteps, nsubmax // nsub)
            ex = comp_dfluxes(fx, dfx, mod.dfluxes[:-1])
            for prob in REPORT_PROBS:
                cn = f"dfx_log10errmax{prob:02d}_nsubsteps{nsub:03d}[-]"
                le = math.log10(ex[prob]) if ex[prob] > 0 else np.nan
                results.loc[irepeat, cn] = le

        import pdb; pdb.set_trace()

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
        opm.context["nuh"] = 11
        task.nreaches = 3
        task.findiff_log10eps = -3
        task.gamma_mean = (1000, 500, 1000)

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
