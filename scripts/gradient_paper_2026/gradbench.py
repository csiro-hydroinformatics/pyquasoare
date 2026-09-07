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

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.stat import armodels

from pyquasoare import gradients

import importlib.util
FTESTS = Path(__file__).parent.parent.parent / "tests"
spec = importlib.util.spec_from_file_location("data_reader", FTESTS / "data_reader.py")
data_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_reader)

np.random.seed(5446)

MODEL_TIMESTEP = 1.

REPORT_PROBS = [0, 50, 90]

DIFF_ORDERS = [1, 2, 3]

def get_options():
    timestep = "daily"
    # Duration depends on number of reaches
    duration_base = {1: 100, 2: 100, 3: 50, 5: 20, 20: 5, 50: 5}
    duration_repeat = 1

    nuh = 5

    stationids = data_reader.SITEIDS

    nreaches_ref = 5
    nreaches = [1, nreaches_ref, 50]

    nalphas_ref = 100
    nalphas = [10, nalphas_ref, 500]

    findiff_log10eps_ref = -5
    findiff_log10eps = [-4, findiff_log10eps_ref, -6]

    nrepeat = 20
    nbatch = 1
    batches = np.arange(nbatch)

    gamma_means = [
        (500, 50, 500),
        (5000, 500, 5000),
        (20, 2, 20)
        ]

    gamma_std = 0.2
    demand_mean = -0.5
    demand_std = 0.2

    opm = hyruns.OptionManager(duration_base=duration_base,
                               duration_repeat=duration_repeat,
                               gamma_std=gamma_std,
                               timestep=timestep,
                               demand_mean=demand_mean,
                               demand_std=demand_std,
                               nrepeat=nrepeat,
                               nuh=nuh,
                               nbatch=nbatch)

    opm.from_cartesian_product(batch=batches,
                               stationid=stationids,
                               nreaches=nreaches,
                               nalphas=nalphas,
                               findiff_log10eps=findiff_log10eps,
                               gamma_mean=gamma_means)
    #tasks = []
    #count = 0
    #for taskid in range(opm.ntasks):
    #    task = opm.get_task(taskid)

    #    # Ensure that at least 2 reference options are met
    #    nre = task.nreaches
    #    nal = task.nalphas
    #    eps = task.findiff_log10eps
    #    match = int(nal == nalphas_ref) \
    #            + int(eps == findiff_log10eps_ref)
    #    if match >= 1:
    #        tasks.append(task.options)
    #opm.tasks = tasks
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


def get_data(config, script_paths, logger):
    stationid = config.task.stationid
    logger.info(f"Load data for stations {stationid}")
    stations = data_reader.get_stations()
    area = stations.loc[stationid, "CATCHMENTAREA[km2]"]

    timestep = config.options.context["timestep"]
    hydroclim = data_reader.get_data(stationid, timestep, nval=None)
    hydroclim = hydroclim.rename(columns={
        "RAINFALL[mm_day-1]": "P",
        "PET[mm_day-1]": "PET"
        })
    hydroclim = hydroclim.loc[:, ["P", "PET"]]

    # Replicate data
    nreaches = config.task.nreaches
    if config.debug:
        nval = 5000
    else:
        nval = int(config.options.context["duration_base"][nreaches] * 365.25)
    hydroclim = hydroclim.iloc[-nval:]
    end = hydroclim.index[-1]
    cc = hydroclim.columns

    nreps = config.options.context["duration_repeat"]
    hydroclim = np.tile(hydroclim, (nreps, 1))
    start = end - pd.DateOffset(days=len(hydroclim) - 1)
    time = pd.date_range(start, end, freq="D")
    hydroclim = pd.DataFrame(hydroclim, index=time,
                             columns=cc)

    data = namedtuple("Data", ["hydroclim", "area"])(hydroclim, area)
    logger.info(".. done")

    return data


def get_reservoir_id(config, ireach, rtype):
    nuh = config.options.context["nuh"]
    nreaches = config.task.nreaches
    nres_per_reach = 5 + nuh

    ibase = nres_per_reach * ireach
    if rtype == "openwater":
        return ibase
    elif rtype == "gr4jprod":
        return ibase + 1
    elif re.search("uh", rtype):
        iuh = int(re.sub("uh", rtype))
        if iuh > nuh - 1:
            raise ValueError(f"Can't have iuh > {nuh - 1}")
        return ibase + 2 + iuh
    elif rtype == "gr4jroute":
        return ibase + 2 + nuh
    elif rtype == "quadrouting":
        return ibase + 3 + nuh
    elif rtype == "transluent":
        return ibase + 4 + nuh
    else:
        raise ValueError(f"Can't recognise rtype {rtype}")


def configure_model(logger, config, area):
    logger.info(f"Create and configure model")
    nuh = config.options.context["nuh"]
    nreaches = config.task.nreaches
    nres_per_reach = 5 + nuh
    nres = nres_per_reach * nreaches
    nalphas = 500 if config.debug else config.task.nalphas
    mod = gradients.MultiReservoirModel(nres, nalphas,
                                        errors="raise")
    mod.area = area

    ires = 0
    for ireach in range(nreaches):
        ibase = nres_per_reach * ireach
        # Transluent = ibase + 4 + nuh

        # RR intercept = ibase
        mod.set_coefs_linear(ibase, "openwatersmooth")

        # RR prod = ibase + 1
        mod.set_coefs_linear(ibase + 1, "gr4jprod")
        mod.add_link_bypass_shortfall(ibase, ibase + 1)

        # RR uh = ibase + 2 .. ibase + 1 + nuh
        for iuh in range(nuh):
            ires = ibase + 2 + iuh
            mod.set_coefs_linear(ires, "linrouting")
            mod.add_link_outflow(ires - 1, ires)

        # RR routing = ibase + 2 + nuh
        igr = ibase + 2 + nuh
        mod.set_coefs_linear(igr, "gr4jroute")
        mod.add_link_outflow(igr - 1, igr)

        # RO route  = ibase + 3 + nuh
        iro = ibase + 3 + nuh
        mod.set_coefs_linear(iro, "quadrouting")

        # Transluent = ibase + 4 + nuh
        itrans = ibase + 4 + nuh
        mod.set_coefs_linear(itrans, "transluent")
        mod.add_link_outflow(iro, itrans)
        conv = 1. / (ireach + 1) # conversion mm subcatch -> mm catch
        mod.add_link_outflow(igr, itrans, conv)

        if ireach > 0:
            # Transluent upstream -> RO route
            mod.add_link_outflow(itrans_up, iro)

        itrans_up = itrans

    mod.nreaches = nreaches
    mod.nres_per_reach = nres_per_reach
    mod.nuh = nuh
    logger.info(".. done")

    return mod


def allocate_model(logger, mod, hydroclim, grad=True):
    nval = len(hydroclim)
    nreaches = mod.nreaches
    nres = mod.nreservoirs
    nres_per_reach = mod.nres_per_reach

    # .. gradients against GR4J X1/X3 + routing capacities
    if grad:
        vspecs = []
        for ireach in range(nreaches):
            ipr = get_reservoir_id(config, ireach, "gr4jprod")
            vspecs.append([ipr, 0]) # Capacity

            irr = get_reservoir_id(config, ireach, "gr4jroute")
            vspecs.append([irr, 0]) # Capacity
    else:
        vspecs = None

    mod.allocate(nval, timestep=MODEL_TIMESTEP,
                 variables_specs=vspecs)

def log_model(logger, mod):
    nreaches = mod.nreaches
    logger.info("****************************************", nret=1)
    logger.info(f"Model description")
    logger.info(f"nreaches   : {mod.nreaches}", ntab=1)
    logger.info(f"nuh        : {mod.nuh}", ntab=1)
    logger.info(f"ntimesteps : {mod.ntimesteps} / {mod.ntimesteps / 365:0.0f}y", ntab=1)
    logger.info(f"nvars      : {len(mod.variables_reservoir_id)}", ntab=1)
    logger.info(f"timestep   : {mod.timestep}", ntab=1)

    gammas = mod.gammas
    fxs = mod.flux_scalings

    for ireach in range(nreaches):
        logger.info(f"Reach {ireach}", nret=1)

        ipr = get_reservoir_id(config, ireach, "gr4jprod")
        logger.info(f"Gamma RR prod = {gammas[ipr]:0.1f}", ntab=1)

        irr = get_reservoir_id(config, ireach, "gr4jroute")
        logger.info(f"Gamma RR route = {gammas[irr]:0.1f}", ntab=1)
        logger.info(f"Demand RR route = {fxs[irr, 1]:0.2f}", ntab=1)

        iro = get_reservoir_id(config, ireach, "quadrouting")
        logger.info(f"Gamma RO route = {gammas[iro]:0.1f}", ntab=1)


def set_qstars(logger, mod, hydroclim):
    mod.Pmean = hydroclim[:, 0].mean()

    # Set 0 flow inputs for first reach
    nval = len(hydroclim)
    Qstars0 = np.zeros((nval, 2))
    iro = get_reservoir_id(config, 0, "quadrouting")
    mod.set_Qstars(iro, Qstars0)

    Qstars1 = np.nan * np.zeros((nval, 2))
    Qstars1[:, 1] = 1.

    # Transform inputs
    tP = forwardP(hydroclim[:, 0])
    rhoP = pd.Series(tP).autocorr()
    alphaP = math.sqrt(1 - rhoP**2) * tP.std() / 10

    tE = forwardE(hydroclim[:, 1])
    rhoE = pd.Series(tE).autocorr()
    alphaE = math.sqrt(1 - rhoE**2) * tE.std() / 10

    # Set inputs
    for ireach in range(mod.nreaches):
        # Reach climate + noise
        # .. rainfall
        nu = np.random.normal(scale=rhoP, size=nval)
        errP = armodels.armodel_sim(alphaP, nu)
        Qstars0[:, 0] = backwardP(tP + errP)

        # .. PET (negative -> demand!)
        nu = np.random.normal(scale=rhoE, size=nval)
        errE = 1 + armodels.armodel_sim(alphaE, nu)
        Qstars0[:, 1] = -backwardE(tE + errE)

        ipr = get_reservoir_id(config, ireach, "openwater")
        mod.set_Qstars(ipr, Qstars0)

        # Reach loss qstar
        irr = get_reservoir_id(config, ireach, "gr4jroute")
        mod.set_Qstars(irr, Qstars1)


def set_parameters(logger, config, mod):
    # Model charac
    nreaches = mod.nreaches
    nrp = mod.nres_per_reach
    nres = nrp * nreaches # 3 reservoirs per reaches

    # Param moments
    gamma_mean = config.task.gamma_mean
    gamma_std = config.options.context["gamma_std"]

    demand_mean = config.options.context["demand_mean"]
    demand_std = config.options.context["demand_std"]

    gammas = 1e-2 * np.ones(nres)
    for ireach in range(nreaches):
        # RR prod reservoir
        ipr = get_reservoir_id(config, ireach, "gr4jprod")
        gammas[ipr] = random_scalings(gamma_std) * gamma_mean[0]

        # RR route reservoir
        irr = get_reservoir_id(config, ireach, "gr4jroute")
        gammas[irr] = random_scalings(gamma_std) * gamma_mean[1]

        dem = np.random.normal(loc=demand_mean, scale=demand_std)
        mod.flux_scalings[irr, 1] = dem

        # RO reservoir
        iro = get_reservoir_id(config, ireach, "quadrouting")
        gammas[iro] = random_scalings(gamma_std) * gamma_mean[2]

    mod.gammas = gammas


def forwardP(x):
    return np.log(x + 1)


def backwardP(y):
    return np.maximum(np.exp(y) - 1, 0.)


def forwardE(x):
    return x


def backwardE(y):
    return np.maximum(y, 0.)


def random_scalings(sigma, nval=1, s0=1e-2):
    if nval == 1:
        return max(s0, 1 + np.random.normal(scale=sigma))
    else:
        return np.maximum(s0, 1 + np.random.normal(scale=sigma, size=nval))


def flux_denom(fx, eps1=1e-4, eps2=1e-3):
    afx = np.abs(fx)
    means = np.mean(afx, axis=0)[None, :]
    maxs = np.max(afx, axis=0)[None, :]
    denom = eps1 + eps2 * means
    denom[maxs < eps1] = np.nan
    return denom

def flux_isabove(afx, prob):
    thresh = np.percentile(afx, prob, axis=0)[None, :]
    return np.where(afx > thresh, 1., np.nan)

def comp_dfluxes(fx_ref, dfx_ref, dfx_alt, eps1=1e-4, eps2=1e-3):
    num = np.abs(dfx_ref - dfx_alt)
    denom = flux_denom(fx_ref, eps1, eps2)
    err = num / denom

    emean = pd.Series(np.nan, index=REPORT_PROBS)
    afx = np.abs(fx_ref)
    for prob in REPORT_PROBS:
        is_above = flux_isabove(afx, prob)[:, None, :, :]
        err_above = err * is_above

        with warnings.catch_warnings():
            # removes nan mean warning
            warnings.simplefilter("ignore")
            em = np.nanmean(err_above, axis=0)

        em[np.isnan(em)] = 0.
        emean[prob] = em.max()

    return emean


def process(config, script_paths, logger, data):
    # Prepare output folders
    fperf = script_paths.fout / "performance"
    fperf.mkdir(exist_ok=True)

    fsims = script_paths.fout / "simulations"
    fsims.mkdir(exist_ok=True)

    # Setup up model
    modref = configure_model(logger, config, data.area)

    # .. difference models
    modneutral = configure_model(logger, config, data.area)

    modeps = {delta: configure_model(logger, config, data.area)
              for delta in [1, 2, 3, 4, 5]}

    hclim = data.hydroclim.values

    allocate_model(logger, modref, hclim)
    allocate_model(logger, modneutral, hclim, False)

    for e, mod in modeps.items():
        allocate_model(logger, mod, hclim, False)

    nruntimeref = modref.ntimesteps / 365.25 * modref.nreservoirs

    # Define job
    batch = config.task.batch
    nbatch = config.options.context["nbatch"]
    nrepeat = 1 if config.debug else config.options.context["nrepeat"]
    ibatches = np.array_split(np.arange(nrepeat), nbatch)[batch]

    # initialise perf results
    mnames = [f"diff{o}" for o in DIFF_ORDERS]
    cc = [f"log10runtime_{m}[sec]" for m in ["ref"] + mnames]
    cc = [f"dfx_log10errmax{p:02d}_{m}[-]" for m in mnames
          for p in REPORT_PROBS]
    results = pd.DataFrame(np.nan, index=ibatches, columns=cc)

    # repeat experiment
    logger.info("")
    for irepeat in ibatches:
        logger.info(f"## Repeat #{irepeat + 1} / {len(ibatches)} ##",
                    nret=2)

        # Set random model inputs
        set_qstars(logger, modref, hclim)
        modneutral.Qstars = modref.Qstars
        for e, mod in modeps.items():
            mod.Qstars = modref.Qstars

        # Set parameters
        set_parameters(logger, config, modref)
        if irepeat == 0:
            log_model(logger, modref)

        modneutral.gammas = modref.gammas.copy()
        modneutral.flux_scalings = modref.flux_scalings.copy()

        for e, mod in modeps.items():
            mod.gammas = modref.gammas.copy()
            mod.flux_scalings = modref.flux_scalings.copy()

        # Run ref model
        logger.info("Run Quasoare - grad model", nret=1)
        t0 = time.time()
        modref.run()
        dt_quas_ref = time.time() - t0

        dt = np.log10(dt_quas_ref / nruntimeref)
        results.loc[irepeat, f"log10runtime_ref[sec]"] = dt

        # Rescale fluxes -> Get dfluxes in mm / d
        nvars = len(modref.variables_reservoir_id)
        for ivar in range(nvars):
            res = modref.variables_reservoir_id[ivar]
            modref.dfluxes[:, ivar] *= modref.gammas[res]

        # Print one simulation to make sure it makes sense
        if irepeat == 0 and modref.nreaches == 5 \
                and config.task.nalphas == 100 \
                and config.task.findiff_log10eps == -5:
            logger.info("Plotting first simul", nret=1)
            nreaches = modref.nreaches
            nval = min(2000, modref.ntimesteps)
            kk = np.arange(-nval, 0)
            nsims = 4
            df = np.empty((nval, nreaches * nsims))
            cols = []
            for ireach in range(nreaches):
                # Data
                iow = get_reservoir_id(config, ireach, "openwater")
                df[:, ireach * nsims] = modref.Qstars[kk, iow, 0]
                df[:, ireach * nsims + 1] = modref.Qstars[kk, iow, 1]

                irr = get_reservoir_id(config, ireach, "gr4jroute")
                qrr = -modref.fluxes[kk, irr, 5]
                df[:, ireach * nsims + 2] = qrr

                irt = get_reservoir_id(config, ireach, "transluent")
                qagg = -modref.fluxes[kk, irt, 5]
                df[:, ireach * nsims + 3] = qagg

                r = f"R{ireach:03d}"
                cols.extend([f"{r}_P", f"{r}_PE", f"{r}_RO_ROUTE",
                             f"{r}_AGG"])

            df = pd.DataFrame(df).round(3)
            df.columns = cols
            df.index.name = "DAY"
            fs = fsims / f"simulations_v{version}_TASK{taskid}.csv"
            csv.write_csv(df, fs, "Simulations for repeat 0",
                          script_paths.source_file,
                          write_index=True)

            plt.close("all")
            fig, axs = plt.subplots(nrows=2, ncols=2,
                                    sharex=True, sharey=True,
                                    figsize=(12, 10), layout="tight")
            for ireach, ax in enumerate(axs.flat):
                if ireach >= nreaches:
                    ax.axis("off")
                    continue

                r = f"R{ireach:03d}"
                df.loc[:, f"{r}_RO_ROUTE"].plot(ax=ax, label="RO-Route")
                df.loc[:, f"{r}_AGG"].plot(ax=ax, label="Agg")

                ax.set_yscale("asinh")
                ax.set_title(f"Reach {ireach}")
                ax.legend(loc=2)

            fig.savefig(fs.parent / f"{fs.stem}.png")

        # --------------------------------------------------
        findiff_eps = 10**(config.task.findiff_log10eps)
        logger.info(f"Numdiff eps = {findiff_eps:0.3e}", nret=1)

        for order in DIFF_ORDERS:
            logger.info(f"Run numerical difference - order {order}", nret=1)

            dfluxes = np.empty((modref.ntimesteps, nvars,
                                modref.nreservoirs,
                                gradients.NFLUXES_OUTPUTS))

            t0 = time.time()
            modneutral.run()
            for ivar, vinfo in modref.variables_specs.iterrows():
                res = vinfo.reservoir_id

                # Run models
                for ieps in np.arange(1, 1 + order):
                    mod = modeps[ieps]
                    mod.gammas[res] = modref.gammas[res] * (1 + ieps * findiff_eps)
                    mod.run()

                # Compute flux gradient
                if order == 1:
                    dfluxes[:, ivar] = (
                        modeps[1].fluxes
                        - modneutral.fluxes) / findiff_eps
                elif order == 2:
                    dfluxes[:, ivar] = (
                        - 1./ 2 * modeps[2].fluxes
                        + 2. * modeps[1].fluxes
                        - 3. / 2 * modneutral.fluxes) / findiff_eps

                elif order == 3:
                    dfluxes[:, ivar] = (
                        1./ 3 * modeps[3].fluxes
                        - 3./ 2 * modeps[2].fluxes
                        + 3. * modeps[1].fluxes
                        - 11. / 6 * modneutral.fluxes) / findiff_eps

                elif order == 4:
                    dfluxes[:, ivar] = (
                        -1./ 4 * modeps[4].fluxes
                        + 4./ 3 * modeps[3].fluxes
                        - 3. * modeps[2].fluxes
                        + 4. * modeps[1].fluxes
                        - 25. / 12 * modneutral.fluxes) / findiff_eps

                # Reset
                for ieps in np.arange(1, 1 + order):
                    mod = modeps[ieps]
                    mod.gammas[res] = modref.gammas[res]

            dt_quas_diff = time.time() - t0

            # Store perf
            dt = np.log10(dt_quas_diff / nruntimeref)
            results.loc[irepeat, f"log10runtime_diff{order}[sec]"] = dt

            ex = comp_dfluxes(modref.fluxes, modref.dfluxes, dfluxes)
            for prob in REPORT_PROBS:
                cn = f"dfx_log10errmax{prob:02d}_diff{order}[-]"
                le = math.log10(ex[prob]) if ex[prob] > 0 else np.nan
                results.loc[irepeat, cn] = le

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

    Config = namedtuple("Config",
                        ["version", "taskid", "overwrite",
                         "debug", "options", "task"])
    opm = get_options()
    task = opm.get_task(taskid)

    if debug:
        opm.context["nuh"] = 11
        task.nreaches = 3
        task.findiff_log10eps = -3
        task.gamma_mean = (1000, 500, 1000)

    config = Config(version, taskid, overwrite,
                    debug, opm, task)

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
