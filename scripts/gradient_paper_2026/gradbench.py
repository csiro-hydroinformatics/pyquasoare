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
from itertools import product as prod
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

import importlib
importlib.reload(gradients)

from utils import get_parameters, forwardP, backwardP
from utils import forwardE, backwardE, random_scalings

import importlib.util
FTESTS = Path(__file__).parent.parent.parent / "tests"
spec = importlib.util.spec_from_file_location("data_reader", FTESTS / "data_reader.py")
data_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_reader)

np.random.seed(5446)

MODEL_TIMESTEP_AGG = 1.

DIFF_ORDERS = [1, 2, 3]

REPORT_PROBS = gradients.compare(np.random.uniform(size=(10, 2, 2)),
                                 np.random.uniform(size=(10, 2, 2)))[0].index

def get_options():
    timestep = "daily"
    # Duration depends on number of reaches
    duration_base = {1: 100, 2: 100, 3: 50, 5: 20, 20: 5, 50: 5}
    duration_repeat = 1

    nuh = 5

    stationids = data_reader.SITEIDS

    nreaches_ref = 5
    nreaches = [1, nreaches_ref, 20]

    nalphas_ref = 50
    nalphas = [10, nalphas_ref, 500]

    findiff_log10eps_ref = -5
    findiff_log10eps = [-4, findiff_log10eps_ref, -6]

    nrepeat = 50
    nbatch = 1
    batches = np.arange(nbatch)

    X0, X1, X2, X3, X4 = get_parameters()
    gamma_means = [
        (X1, X3, X1),
        (X1 * 10, X3 * 10, X1 * 10),
        (X1 / 10, X3 / 10, X1 / 10)
        ]

    gamma_std = 0.5
    demand_mean = -0.5
    demand_std = 0.5

    x4_mean = 1.
    x4_std = 0.5

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
                               nalphas_ref=nalphas_ref,
                               findiff_log10eps_ref=findiff_log10eps_ref,
                               nreaches_ref=nreaches_ref,
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
        ntimestep_agg = 5000
    else:
        ntimestep_agg = int(config.options.context["duration_base"][nreaches] * 365.25)
    hydroclim = hydroclim.iloc[-ntimestep_agg:]
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
        iuh = int(re.sub("uh", "", rtype))
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


def configure_model(config, logger, area):
    logger.info(f"Create and configure model")
    nsubsteps = config.nsubsteps
    nuh = config.options.context["nuh"]
    nreaches = config.task.nreaches
    nres_per_reach = 5 + nuh
    nres = nres_per_reach * nreaches
    nalphas = config.task.nalphas
    mod = gradients.MultiReservoirModel(nres,
                                        nalphas=nalphas,
                                        nsubsteps=nsubsteps,
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


def allocate_model(config, logger, mod, hydroclim, grad=True):
    ntimesteps_agg = len(hydroclim)
    nreaches = mod.nreaches

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

    mod.allocate(ntimesteps_agg,
                 timestep_agg=MODEL_TIMESTEP_AGG,
                 variables_specs=vspecs)

def log_model(config, logger, mod):
    nreaches = mod.nreaches
    logger.info("****************************************", nret=1)
    logger.info(f"Model description")
    logger.info(f"nreaches       : {mod.nreaches}", ntab=1)
    logger.info(f"nalphas        : {mod.nalphas}", ntab=1)
    logger.info(f"nuh            : {mod.nuh}", ntab=1)
    logger.info(f"nsubsteps      : {mod.nsubsteps}", ntab=1)
    ny = mod.ntimesteps_agg / 365
    logger.info(f"ntimesteps_agg : {mod.ntimesteps_agg} ({ny:0.0f}y)", ntab=1)
    logger.info(f"nvars          : {len(mod.variables_reservoir_id)}", ntab=1)
    logger.info(f"timestep_agg   : {mod.timestep_agg:0.3f}", ntab=1)

    gammas = mod.gammas
    fxs = mod.flux_scalings
    nres_per_reach = mod.nres_per_reach

    for ireach in range(nreaches):
        res1 = ireach * nres_per_reach
        res2 = res1 + nres_per_reach - 1
        logger.info(f"Reach {ireach} (reservoirs {res1}-{res2})", nret=1)

        ipr = get_reservoir_id(config, ireach, "gr4jprod")
        logger.info(f"Gamma RR prod = {gammas[ipr]:0.1f}", ntab=1)

        irr = get_reservoir_id(config, ireach, "gr4jroute")
        logger.info(f"Gamma RR route = {gammas[irr]:0.1f}", ntab=1)
        logger.info(f"Demand RR route = {fxs[irr, 1]:0.2f}", ntab=1)

        iro = get_reservoir_id(config, ireach, "quadrouting")
        logger.info(f"Gamma RO route = {gammas[iro]:0.1f}", ntab=1)

        iru = get_reservoir_id(config, ireach, f"uh0")
        x4 = 1.25 * mod.nuh / mod.coefs_linear[iru, 2, 0, 0]
        logger.info(f"uh x4 = {x4:0.1f}", ntab=1)


def set_qstars(config, logger, mod, hydroclim, perturb=True):
    nsubsteps = config.nsubsteps

    # Set 0 flow inputs for first reach
    ntimesteps_agg = len(hydroclim)
    Qstars0 = np.zeros((ntimesteps_agg, 2))
    iro = get_reservoir_id(config, 0, "quadrouting")
    mod.set_Qstars_one_reservoir(iro, Qstars0)

    Qstars1 = np.nan * np.zeros((ntimesteps_agg, 2))
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
        if not perturb:
            Qstars0[:, 0] = backwardP(tP)
            Qstars0[:, 1] = -backwardE(tE)
        else:
            # Reach climate + noise
            # .. rainfall
            nu = np.random.normal(scale=rhoP, size=ntimesteps_agg)
            errP = armodels.armodel_sim(alphaP, nu)
            Qstars0[:, 0] = backwardP(tP + errP)

            # .. PET (negative -> demand!)
            nu = np.random.normal(scale=rhoE, size=ntimesteps_agg)
            errE = 1 + armodels.armodel_sim(alphaE, nu)
            Qstars0[:, 1] = -backwardE(tE + errE)

        ipr = get_reservoir_id(config, ireach, "openwater")
        mod.set_Qstars_one_reservoir(ipr, Qstars0)

        # Reach loss qstar
        irr = get_reservoir_id(config, ireach, "gr4jroute")
        mod.set_Qstars_one_reservoir(irr, Qstars1)


def set_parameters(config, logger, mod):
    # Model charac
    nreaches = mod.nreaches
    nrp = mod.nres_per_reach
    nres = nrp * nreaches # 3 reservoirs per reaches

    # Param moments
    gamma_mean = config.task.gamma_mean
    gamma_std = config.options.context["gamma_std"]

    demand_mean = config.options.context["demand_mean"]
    demand_std = config.options.context["demand_std"]

    x4_mean = config.options.context["x4_mean"]
    x4_std = config.options.context["x4_std"]

    gammas = np.ones(nres)
    nuh = config.options.context["nuh"]
    for ireach in range(nreaches):
        # RR prod reservoir
        ipr = get_reservoir_id(config, ireach, "gr4jprod")
        gammas[ipr] = max(1, random_scalings(gamma_std) * gamma_mean[0])

        # RR route reservoir
        irr = get_reservoir_id(config, ireach, "gr4jroute")
        gammas[irr] = max(1, random_scalings(gamma_std) * gamma_mean[1])

        dem = np.random.normal(loc=demand_mean, scale=demand_std)
        mod.flux_scalings[irr, 1] = min(0, dem)

        # RO reservoir
        iro = get_reservoir_id(config, ireach, "quadrouting")
        gammas[iro] = max(1, random_scalings(gamma_std) * gamma_mean[2])

        # UH
        x4 = max(0.05, random_scalings(x4_std) * x4_mean)
        for iuh in range(nuh):
            iru = get_reservoir_id(config, ireach, f"uh{iuh}")
            mod.coefs_linear[iru, 2, :, 0] = 1.25 * nuh / x4

    # Store reservoir capacities
    mod.gammas = gammas



def process(config, script_paths, logger, data):
    # Prepare output folders
    fperf = script_paths.fout / "performance"
    fperf.mkdir(exist_ok=True)

    fsims = script_paths.fout / "simulations"
    fsims.mkdir(exist_ok=True)

    # Setup up model
    modref = configure_model(config, logger, data.area)

    # .. difference models
    modneutral = configure_model(config, logger, data.area)

    modeps = {delta: configure_model(config, logger, data.area)
              for delta in [1, 2, 3]}

    hclim = data.hydroclim.values

    allocate_model(config, logger, modref, hclim)
    allocate_model(config, logger, modneutral, hclim, False)

    for e, mod in modeps.items():
        allocate_model(config, logger, mod, hclim, False)

    nruntimeref = modref.ntimesteps_agg / 365.25 * modref.nreservoirs

    # Define job
    batch = config.task.batch
    nbatch = config.options.context["nbatch"]
    nrepeat = 1 if config.debug else config.options.context["nrepeat"]
    ibatches = np.array_split(np.arange(nrepeat), nbatch)[batch]

    # initialise perf results
    mnames = [f"diff{o}" for o in DIFF_ORDERS]
    cc = [f"log10runtime_{m}[sec]" for m in ["ref", "neutral"] + mnames]
    cc += [f"log10runtimedeltaneutral_{m}[sec]" for m in ["ref"] + mnames]

    cc += [f"dfx_log10errmax{p:02d}_{m}[-]" for m in mnames
           for p in REPORT_PROBS]
    results = pd.DataFrame(np.nan, index=ibatches, columns=cc)

    results.loc[:, "ntimesteps_agg"] = modref.ntimesteps_agg
    results.loc[:, "nreservoirs"] = modref.nreservoirs
    results.loc[:, "nvariables"] = len(modref.variables_reservoir_id)

    # repeat experiment
    logger.info("")
    for irepeat in ibatches:
        logger.info(f"## Repeat #{irepeat + 1} / {len(ibatches)} ##",
                    nret=2)

        # Set random model inputs
        set_qstars(config, logger, modref, hclim)
        k = np.arange(0, modref.ntimesteps, modref.nsubsteps)
        modneutral.set_Qstars(modref.Qstars[k])
        for e, mod in modeps.items():
            mod.set_Qstars(modref.Qstars[k])

        # Set parameters
        set_parameters(config, logger, modref)
        if irepeat == 0:
            log_model(config, logger, modref)

        modneutral.gammas = modref.gammas.copy()
        modneutral.flux_scalings = modref.flux_scalings.copy()
        modneutral._coefs_linear = modref._coefs_linear.copy()

        for e, mod in modeps.items():
            mod.gammas = modref.gammas.copy()
            mod.flux_scalings = modref.flux_scalings.copy()
            mod._coefs_linear = modref._coefs_linear.copy()

        # Run ref model
        logger.info("Run Quasoare - grad model", nret=1)

        t0 = time.time()
        modneutral.run()
        modneutral.aggregate()
        dt_quas_neutral = time.time() - t0

        dt = np.log10(dt_quas_neutral / nruntimeref)
        results.loc[irepeat, f"log10runtime_neutral[sec]"] = dt

        t0 = time.time()
        modref.run()
        modref.aggregate()
        dt_quas_ref = time.time() - t0

        dt = np.log10(dt_quas_ref / nruntimeref)
        results.loc[irepeat, "log10runtime_ref[sec]"] = dt

        delta = np.log10(dt_quas_ref) - np.log10(dt_quas_neutral)
        results.loc[irepeat, "log10runtimedeltaneutral_ref[sec]"] = delta

        # Rescale fluxes -> Get dfluxes in mm / d
        nvars = len(modref.variables_reservoir_id)
        for ivar, res in enumerate(modref.variables_reservoir_id):
            modref.dfluxes_agg[:, ivar] *= modref.gammas[res]

        # Print one simulation to make sure it makes sense
        if irepeat == 0 and ((modref.nreaches == 5 \
                and config.task.nalphas == 100 \
                and config.task.findiff_log10eps == -5) or config.debug):
            logger.info("Plotting first simul", nret=1)
            nreaches = modref.nreaches
            ntimestep_agg = min(2000, modref.ntimesteps)
            kk = np.arange(-ntimestep_agg, 0)
            nsims = 5
            df = np.empty((ntimestep_agg, nreaches * nsims))
            cols = []
            for ireach in range(nreaches):
                # Data
                iow = get_reservoir_id(config, ireach, "openwater")
                df[:, ireach * nsims] = modref.Qstars[kk, iow, 0]
                df[:, ireach * nsims + 1] = modref.Qstars[kk, iow, 1]

                irp = get_reservoir_id(config, ireach, "gr4jprod")
                qrp = -modref.fluxes[kk, irp, 5]
                df[:, ireach * nsims + 2] = qrp

                irr = get_reservoir_id(config, ireach, "gr4jroute")
                qrr = -modref.fluxes[kk, irr, 5]
                df[:, ireach * nsims + 3] = qrr

                irt = get_reservoir_id(config, ireach, "transluent")
                qagg = -modref.fluxes[kk, irt, 5]
                df[:, ireach * nsims + 4] = qagg

                r = f"R{ireach:03d}"
                cols.extend([f"{r}_P", f"{r}_PE",
                             f"{r}_RR_PROD",
                             f"{r}_RO_ROUTE",
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
                df.loc[:, f"{r}_RR_PROD"].plot(ax=ax, label="RR-Prod")
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

            dfluxes = np.empty((modref.ntimesteps_agg, nvars,
                                modref.nreservoirs,
                                gradients.NFLUXES_OUTPUTS))

            t0 = time.time()
            modneutral.run()
            modneutral.aggregate()

            for ivar, vinfo in modref.variables_specs.iterrows():
                res = vinfo.reservoir_id

                # Run models
                for ieps in np.arange(1, 1 + order):
                    mod = modeps[ieps]
                    # .. set param change
                    g0 = modref.gammas[res]
                    g1 = g0 * (1 + ieps * findiff_eps)
                    mod.gammas[res] = g1
                    # .. run model
                    mod.run()
                    mod.aggregate()
                    # ..reset back to original
                    mod.gammas[res] = g0

                # Compute flux gradient
                # we divide by findiff_eps only, hence the gradient is
                # Γ df/dΓ and not df/dΓ
                if order == 1:
                    dfluxes[:, ivar] = (
                        modeps[1].fluxes_agg
                        - modneutral.fluxes_agg) / findiff_eps

                elif order == 2:
                    dfluxes[:, ivar] = (
                        - 1./ 2 * modeps[2].fluxes_agg
                        + 2. * modeps[1].fluxes_agg
                        - 3. / 2 * modneutral.fluxes_agg) / findiff_eps

                elif order == 3:
                    dfluxes[:, ivar] = (
                        1./ 3 * modeps[3].fluxes_agg
                        - 3./ 2 * modeps[2].fluxes_agg
                        + 3. * modeps[1].fluxes_agg
                        - 11. / 6 * modneutral.fluxes_agg) / findiff_eps

            dt_quas_diff = time.time() - t0

            # Store perf
            dt = np.log10(dt_quas_diff / nruntimeref)
            results.loc[irepeat, f"log10runtime_diff{order}[sec]"] = dt

            delta = np.log10(dt_quas_diff) - np.log10(dt_quas_neutral)
            results.loc[irepeat, f"log10runtimedeltaneutral_diff{order}[sec]"] = delta

            ex, st = gradients.compare(modref.dfluxes_agg, dfluxes)
            for prob, val in ex.items():
                cn = f"dfx_log10errmax{prob:02d}_diff{order}[-]"
                le = math.log10(val) if val > 0 else np.nan
                results.loc[irepeat, cn] = le

            #if config.debug and ex.max() > 10.:
            #    imax = np.where(st[90] == st[90].max())
            #    ivar, res, ifx = [ii[0] for ii in imax]

            #    plt.close("all")
            #    fig, axs = plt.subplots(nrows=3, figsize=(10, 8),
            #                            layout="constrained", sharex=True)
            #    ax = axs[0]
            #    nsub = modref.nsubsteps
            #    k = np.arange(nsub, modref.ntimesteps, nsub)
            #    y1 = modref.x[k, res]
            #    ax.plot(y1)

            #    ax = axs[1]
            #    y1 = modref.fluxes_agg[:, res, ifx]
            #    y2 = modeps[1].fluxes_agg[:, res, ifx]
            #    ax.plot(y1)
            #    ax.plot(y2)
            #    ax.set(yscale="asinh")

            #    ax = axs[2]
            #    y1 = modref.dfluxes_agg[:, ivar, res, ifx]
            #    y2 = dfluxes[:, ivar, res, ifx]
            #    ax.plot(y1)
            #    ylim = ax.get_ylim()
            #    ax.plot(y2)
            #    ax.set(yscale="asinh", ylim=ylim)

            #    ftitle = f"Res {res} - Var {ivar} - Fx {ifx}"
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
    nsubsteps = 1

    Config = namedtuple("Config",
                        ["version", "taskid", "overwrite",
                         "debug", "options", "task",
                         "nsubsteps"])
    opm = get_options()
    task = opm.get_task(taskid)

    if debug:
        task.nreaches = 5
        task.nalphas = 50
        task.findiff_log10eps = -5
        _, X1, X2, X3, _ = get_parameters()
        #task.gamma_mean = (X1 / 10., X3 / 10., X1 / 10.)
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
