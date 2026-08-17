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

from hydrodiy.io import csv, iutils, hyruns
from hydrodiy.stat import armodels

from pyquasoare import gradients
from pyquasoare.benchmarks import benchmarks

import importlib.util
FTESTS = Path(__file__).parent.parent.parent / "tests"
spec = importlib.util.spec_from_file_location("data_reader", FTESTS / "data_reader.py")
data_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_reader)

np.random.seed(5446)

MODEL_TIMESTEP = 1.

FLUX_FUNCTIONS = {
    "gr4jprodquad": [
        lambda x: 1 - np.clip(x, 0, 1)**2,
        lambda x: np.clip(x, 0, 1) * (2 - np.clip(x, 0, 1)),
        lambda x: 4**3/9**4 * np.maximum(x, 0)**2
        ],
    "gr4jroutequad": [
        lambda x: 0.9 * np.ones_like(x),
        lambda x: np.maximum(x, 0.)**2,
        lambda x: np.maximum(x, 0.)**2 / 4
        ],
}


def get_options():
    timestep = "daily"
    # Duration depends on number of reaches
    duration_base = {1: 100, 3: 50, 5: 50, 50: 10}
    duration_repeat = 1

    stationids = data_reader.SITEIDS

    nreaches_ref = 5
    nreaches = [1, nreaches_ref, 50]

    nalphas_ref = 100
    nalphas = [10, nalphas_ref, 1000]

    numdiff_eps_ref = 1e-5
    numdiff_eps = [1e-3, numdiff_eps_ref, 1e-7, 1e-9, 1e-11]

    nrepeat = 50
    nbatch = 1
    batches = np.arange(nbatch)

    gamma_means = [
        (500, 50, 1000),
        (5000, 5000, 10000),
        (50, 5, 10)
        ]

    gamma_std = 0.3
    demand_mean = -0.5
    demand_std = 0.3

    opm = hyruns.OptionManager(duration_base=duration_base,
                               duration_repeat=duration_repeat,
                               gamma_std=gamma_std,
                               timestep=timestep,
                               demand_mean=demand_mean,
                               demand_std=demand_std,
                               nrepeat=nrepeat,
                               nbatch=nbatch)

    opm.from_cartesian_product(batch=batches,
                               stationid=stationids,
                               nreaches=nreaches,
                               nalphas=nalphas,
                               numdiff_eps=numdiff_eps,
                               gamma_mean=gamma_means)
    tasks = []
    count = 0
    for taskid in range(opm.ntasks):
        task = opm.get_task(taskid)

        # Ensure that at least 2 reference options are met
        nre = task.nreaches
        nal = task.nalphas
        eps = task.numdiff_eps
        match = int(nre == nreaches_ref) \
                + int(nal == nalphas_ref) \
                + int(eps == numdiff_eps_ref)
        if match >= 2:
            tasks.append(task.options)
    opm.tasks = tasks
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

    nreaches = config.task.nreaches
    nval = config.options.context["duration_base"][nreaches] * 365
    ntot = config.options.context["duration_repeat"] * 365
    nreps = ntot // nval

    timestep = config.options.context["timestep"]
    hydroclim = data_reader.get_data(stationid, timestep, nval=None)
    hydroclim = hydroclim.rename(columns={
        "RAINFALL[mm_day-1]": "P",
        "PET[mm_day-1]": "PET"
        })
    hydroclim = hydroclim.loc[:, ["P", "PET"]]

    data = namedtuple("Data", ["hydroclim", "area"])(hydroclim, area)
    logger.info(".. done")

    return data

def configure_model(logger, config, area):
    logger.info(f"Create and configure model")
    nreaches = config.task.nreaches
    nres_per_reach = 4
    nres = nres_per_reach * nreaches
    nalphas = 1000 if config.debug else config.task.nalphas
    mod = gradients.MultiReservoirModel(nres, nalphas,
                                        errors="raise")
    mod.area = area

    for ireach in range(nreaches):
        ibase = nres_per_reach * ireach
        # RR prod = ibase
        # RR routing = ibase + 1
        # RO routing = ibase + 2
        # Transluent = ibase + 3

        # .. links ...
        # RR prod -> RR routing
        mod.add_link_outflow(ibase, ibase + 1)

        # RR route -> transluent with conversion
        conv = area * MODEL_TIMESTEP / 86.4
        mod.add_link_outflow(ibase + 1, ibase + 3, conv)

        # RO route -> Transluent
        mod.add_link_outflow(ibase + 2, ibase + 3)

        if ireach > 0:
            # Transluent upstream -> RO route
            mod.add_link_outflow(ibase - 1, ibase + 2)

    mod.nreaches = nreaches
    mod.nres_per_reach = nres_per_reach
    logger.info(".. done")

    return mod


def set_coefs_linear(logger, mod):
    logger.info("Setting model coefs")
    nreaches = mod.nreaches
    nres_per_reach = mod.nres_per_reach
    for ireach in range(nreaches):
        ibase = nres_per_reach * ireach
        # RR prod = ibase
        # RR routing = ibase + 1
        # RO routing = ibase + 2
        # Transluent = ibase + 3

        # Set RR prod coefs
        mod.set_coefs_linear(ibase, FLUX_FUNCTIONS["gr4jprodquad"])

        # Set RR route coefs
        mod.set_coefs_linear(ibase + 1, FLUX_FUNCTIONS["gr4jroutequad"])

        # Set RO route coefs
        mod.set_coefs_linear(ibase + 2, "quadrouting")

        # Set Transluent coefs
        mod.set_coefs_linear(ibase + 3, "transluent")
    logger.info(".. done")


def allocate_model(logger, mod, hydroclim, grad=True):
    nval = len(hydroclim)

    # .. gradients against all capacities
    nres = mod.nreservoirs
    nres_per_reach = mod.nres_per_reach
    if grad:
        vspecs = [[res, 0] for res in range(nres)]
    else:
        vspecs = None
    mod.allocate(nval, timestep=MODEL_TIMESTEP,
                 variables_specs=vspecs)


def set_qstars(logger, mod, hydroclim):
    mod.Pmean = hydroclim[:, 0].mean()

    # Set 0 flow inputs for first reach
    nval = len(hydroclim)
    Qstars0 = np.zeros((nval, 2))
    mod.set_Qstars(3, Qstars0)

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
        # Reach input noise
        # .. rainfall
        nu = np.random.normal(scale=rhoP, size=nval)
        errP = armodels.armodel_sim(alphaP, nu)
        Qstars0[:, 0] = backwardP(tP + errP)

        # .. PET (negative -> demand!)
        nu = np.random.normal(scale=rhoE, size=nval)
        errE = 1 + armodels.armodel_sim(alphaE, nu)
        Qstars0[:, 1] = -backwardE(tE + errE)

        ibase = ireach * mod.nres_per_reach
        mod.set_Qstars(ibase, Qstars0)

        # Reach loss qstar
        mod.set_Qstars(ibase + 1, Qstars1)


def set_parameters(logger, config, mod):
    # Model charac
    nreaches = mod.nreaches
    nrp = mod.nres_per_reach
    nres = nrp * nreaches # 3 reservoirs per reaches

    # Param moments
    gamma_mean = config.task.gamma_mean
    gamma_std = config.options.context["gamma_std"]

    gammas = np.zeros(nres)

    # .. RR prod reservoir
    s = random_scalings(gamma_std, nreaches)
    grr_prod = gamma_mean[0] * s
    gammas[::nrp] = grr_prod

    # .. RR route reservoir
    s = random_scalings(gamma_std, nreaches)
    grr_route = gamma_mean[1] * s
    gammas[1::nrp] = grr_route

    # .. RO reservoir -> m3
    factor = mod.Pmean * mod.area * np.arange(1, nreaches + 1)
    factor *= gamma_mean[2] / 86.4
    s = random_scalings(gamma_std, nreaches)
    gro_route = factor * s
    gammas[2::nrp] = gro_route

    # .. transluent capacity
    gammas[3::nrp] = 1.

    mod.gammas = gammas

    # Demand moments
    demand_mean = config.options.context["demand_mean"]
    demand_std = config.options.context["demand_std"]
    dem = demand_mean + demand_std * np.random.standard_normal(size=nreaches)
    mod.flux_scalings[1::nrp, 1] = dem

    return grr_prod, grr_route, gro_route, dem


def forwardP(x):
    return np.log(x + 1)


def backwardP(y):
    return np.maximum(np.exp(y) - 1, 0.)


def forwardE(x):
    return x


def backwardE(y):
    return np.maximum(y, 0.)


def forwardasinh(x, x0=1):
    return np.asinh(x / x0) * x0


def random_scalings(sigma, nval, s0=1e-2):
    return np.maximum(s0, 1 + np.random.normal(scale=sigma, size=nval))


def forward_plot_scale(x, eps=1e-2):
    return (np.sqrt(eps**2 + np.abs(x)) - eps) * np.sign(x)


def backward_plot_scale(y, eps=1e-2):
    return ((np.abs(y) + eps)**2 - eps**2) * np.sign(y)


def compute_flux_error(bench, alt, transform=None, nan="ignore"):
    nreaches = bench.shape[1] // 4
    ires = (np.arange(nreaches)[:, None] + np.array([0, 1, 3])[None, :]).ravel()
    fb = bench[:, ires, 5]
    fa = alt[:, ires, 5]
    if transform is not None:
        fw = forward_asinh if transform=="asinh" else lambda x: x
        fb = fw(fb)
        fa = fw(fa)

    err = np.abs(fa - fb)
    if nan == "keep":
        err[np.isnan(fa) | np.isnan(fb)] = np.nan

    emax = np.nanmax(err)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var = ((fb - np.nanmean(fb, axis=0))**2).sum(axis=0)
        eff = (err**2).sum(axis=0) / var
    coefdet = np.nanmax(eff)

    return emax, coefdet


def compute_dflux_error(dbench, dalt, transform=None, nan="ignore"):
    nreaches = dbench.shape[1] // 4
    ires = (np.arange(nreaches)[:, None] + np.array([0, 1, 3])[None, :]).ravel()

    dfb = dbench[:, ires, ires, 5]
    dfa = dalt[:, ires, ires, 5]
    if transform:
        fw = forward_asinh if transform=="asinh" else lambda x: x
        dfb = fw(dfb)
        dfa = fw(dfa)

    err = np.abs(dfb - dfa)
    if nan == "keep":
        err[np.isnan(dfa) | np.isnan(dfb)] = np.nan

    emax = np.nanmax(err)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        var = ((dfb - np.nanmean(dfb, axis=0))**2).sum(axis=0)
        eff = (err**2).sum(axis=0) / var

    coefdet = np.nanmax(eff)

    return emax, coefdet

def convert_fluxes(area, nres_per_reach, gammas,
                   fluxes, dfluxes):
    nres = dfluxes.shape[2]
    nreaches = nres // nres_per_reach

    fluxes_converted = fluxes.copy()

    # Convert grad against capacity to grad
    # against capacity scaling to maintain unit
    g = gammas[None, None, :, None]
    dfluxes_converted = dfluxes.copy() * g

    # Convert m3/s to mm/day for the transluent reservoir
    for ireach in range(nreaches):
        area_up = area * (ireach + 1)
        ires_transluent = ireach * nres_per_reach + 3
        factor = 86.4 * MODEL_TIMESTEP / area_up

        fluxes_converted[:, ires_transluent] = \
            factor * fluxes[:, ires_transluent]

        dfluxes_converted[:, :, ires_transluent] = \
            factor * dfluxes_converted[:, :, ires_transluent]

    return fluxes_converted, dfluxes_converted


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
    modlow = configure_model(logger, config, data.area)
    modhigh = configure_model(logger, config, data.area)

    # Allocate model with extended hydroclim data
    debug = config.debug
    context = config.options.context
    nreaches = config.task.nreaches
    dbase = 20 if debug else context["duration_base"][nreaches]
    drep = 1 if debug else context["duration_repeat"]
    n = dbase * 365
    hclim = np.tile(data.hydroclim.iloc[-n:], (drep, 1))

    #if debug:
    #    hclim = hclim[:2000]

    allocate_model(logger, modref, hclim)
    allocate_model(logger, modneutral, hclim, False)
    allocate_model(logger, modlow, hclim, False)
    allocate_model(logger, modhigh, hclim, False)

    # Define job
    batch = config.task.batch
    nbatch = config.options.context["nbatch"]
    nrepeat = 3 if config.debug else config.options.context["nrepeat"]
    ibatches = np.array_split(np.arange(nrepeat), nbatch)[batch]

    logger.info("initialise gradient data")
    nreaches = config.task.nreaches
    nres = modref.nreservoirs
    nres_per_reach = modref.nres_per_reach

    dgrr_prod = np.zeros((nres, nreaches))
    k1 = np.arange(nreaches) * nres_per_reach
    k2 = np.arange(nreaches)
    dgrr_prod[k1, k2] = 1

    dgrr_route = np.zeros((nres, nreaches))
    dgrr_route[k1 + 1, k2] = 1

    dgro_route = np.zeros((nres, nreaches))
    dgro_route[k1 + 2, k2] = 1

    areas = np.ones(nreaches) * modref.area
    x0 = 0.5

    logger.info("initialise benchmark data")
    nval = len(hclim)
    x_rr_prod = np.zeros((nval, nreaches))
    dx_rr_prod = np.zeros((nres, nval, nreaches))
    x_rr_route = np.zeros((nval, nreaches))
    x_rr_route = np.zeros((nval, nreaches))
    dx_rr_route = np.zeros((nres, nval, nreaches))
    x_ro_route = np.zeros((nval, nreaches))
    dx_ro_route = np.zeros((nres, nval, nreaches))
    logger.info(".. step 1")

    fx_rr_prod = np.zeros((nval, nreaches))
    fx_rr_prod_p = np.zeros((nval, nreaches))
    fx_rr_prod_m = np.zeros((nval, nreaches))
    dfx_rr_prod = np.zeros((nres, nval, nreaches))
    fx_rr_route = np.zeros((nval, nreaches))
    fx_rr_route_p = np.zeros((nval, nreaches))
    fx_rr_route_m = np.zeros((nval, nreaches))
    dfx_rr_route = np.zeros((nres, nval, nreaches))
    fx_ro_route = np.zeros((nval, nreaches))
    fx_ro_route_p = np.zeros((nval, nreaches))
    fx_ro_route_m = np.zeros((nval, nreaches))
    dfx_ro_route = np.zeros((nres, nval, nreaches))
    logger.info(".. step 2")

    fluxes_bench = modref.fluxes.copy()
    fluxes_bench.fill(np.nan)
    dfluxes_bench = modref.dfluxes.copy()
    dfluxes_bench.fill(np.nan)
    logger.info(".. step 3")

    logger.info("initialise numerical difference data")
    numdiff_eps = 1e-2 if config.debug else config.task.numdiff_eps
    dfluxes_quas_diff1 = modref.dfluxes.copy() * np.nan
    dfluxes_quas_diff2 = modref.dfluxes.copy() * np.nan

    dfluxes_bench_diff1 = modref.dfluxes.copy() * np.nan
    dfluxes_bench_diff2 = modref.dfluxes.copy() * np.nan

    # initialise perf results
    mnames = ["bench", "bench_diff1", "bench_diff2",
              "quasoare", "quasoare_diff1", "quasoare_diff2"]
    cc = [f"runtime_{m}" for m in mnames] + ["runtime_quasoare_neutral"]
    cc += ["fx_errmax_quasoare", "fx_coefdet_quasoare"]
    cc += [f"dfx_errmax_{m}" for m in mnames[1:]]
    cc += [f"dfx_coefdet_{m}" for m in mnames[1:]]
    results = pd.DataFrame(0., index=np.arange(nrepeat),
                           columns=cc)

    # repeat experiment
    logger.info("")
    for irepeat in range(nrepeat):
        logger.info(f"## Repeat #{irepeat + 1} / {nrepeat} ###", nret=2)
        # Set random model inputs
        set_qstars(logger, modref, hclim)
        modneutral.Qstars = modref.Qstars
        modlow.Qstars = modref.Qstars
        modhigh.Qstars = modref.Qstars

        grr_prod, grr_route, gro_route, demrr_route = \
                set_parameters(logger, config, modref)
        logger.info(f"GRR prod = {grr_prod.round(0)}", ntab=1)
        logger.info(f"GRR route = {grr_route.round(0)}", ntab=1)
        logger.info(f"DEM route = {demrr_route.round(2)}", ntab=1)
        logger.info(f"GRO route = {gro_route.round(0)}", ntab=1)

        grr_prod_p = grr_prod.copy()
        grr_prod_m = grr_prod.copy()
        grr_route_p = grr_route.copy()
        grr_route_m = grr_route.copy()
        gro_route_p = gro_route.copy()
        gro_route_m = gro_route.copy()

        # --------------------------------------------------
        # Set models
        modneutral.gammas = modref.gammas
        modneutral.flux_scalings = modref.flux_scalings

        modlow.gammas = modref.gammas
        modlow.flux_scalings = modref.flux_scalings

        modhigh.gammas = modref.gammas
        modhigh.flux_scalings = modref.flux_scalings

        # --------------------------------------------------
        # Run benchmark
        pin = modref.Qstars[:, ::modref.nres_per_reach, 0]
        # .. convert to positive ! -> classical convention in RR
        ein = -modref.Qstars[:, ::modref.nres_per_reach, 1]

        t0 = time.time()
        # .. run tangent code for the capacity of reach reservoir
        for res in np.arange(nres)[::-1]:
            #if res > 0:
            #    continue
            out = x_rr_prod, dx_rr_prod[res], \
                x_rr_route, dx_rr_route[res], \
                x_ro_route, dx_ro_route[res], \
                fx_rr_prod, dfx_rr_prod[res], \
                fx_rr_route, dfx_rr_route[res], \
                fx_ro_route, dfx_ro_route[res], \

            benchmarks.gradbench_semidist_tangent(MODEL_TIMESTEP,
                                                  pin, ein, x0, areas,
                                                  grr_prod,
                                                  dgrr_prod[res],
                                                  grr_route,
                                                  dgrr_route[res],
                                                  gro_route,
                                                  dgro_route[res],
                                                  demrr_route,
                                                  out)
        dt_bench = time.time() - t0

        # .. rearrange benchmark data to match model format
        for res in np.arange(nres):
            ireach = res // nres_per_reach
            ires_in_reach = res % nres_per_reach
            if ires_in_reach == 0:
                # .. fluxes
                fluxes_bench[:, res, 5] = -fx_rr_prod[:, ireach]
                # .. fluxes grad
                for resg in range(nres):
                    dfluxes_bench[:, resg, res, 5] = -dfx_rr_prod[resg, :, ireach]

            elif ires_in_reach == 1:
                # .. fluxes
                fluxes_bench[:, res, 5] = -fx_rr_route[:, ireach]
                # .. fluxes grad
                for resg in range(nres):
                    dfluxes_bench[:, resg, res, 5] = -dfx_rr_route[resg, :, ireach]

            elif ires_in_reach == 3:
                # .. fluxes
                fluxes_bench[:, res, 5] = -fx_ro_route[:, ireach]
                # .. fluxes grad
                for resg in range(nres):
                    dfluxes_bench[:, resg, res, 5] = -dfx_ro_route[resg, :, ireach]

        # --------------------------------------------------
        # Run quasoare gradient
        # .. Modify alphas to match benchmark
        for res in np.arange(nres):
            ireach = res // nres_per_reach
            ires_in_reach = res % nres_per_reach
            if ires_in_reach == 0:
                xmin = x_rr_prod[:, ireach].min(axis=0)
                xmax = x_rr_prod[:, ireach].max(axis=0)
            elif ires_in_reach == 1:
                xmin = x_rr_route[:, ireach].min(axis=0)
                xmax = x_rr_route[:, ireach].max(axis=0)
            elif ires_in_reach == 2:
                xmin = x_ro_route[:, ireach].min(axis=0)
                xmax = x_ro_route[:, ireach].max(axis=0)

            xmin = np.floor(xmin / 0.1) * 0.1
            xmax = max(1., np.ceil(xmax / 0.1) * 0.1)

            modref.set_alphas(res, xmin, xmax)
            modneutral.set_alphas(res, xmin, xmax)
            modlow.set_alphas(res, xmin, xmax)
            modhigh.set_alphas(res, xmin, xmax)

        set_coefs_linear(logger, modref)
        set_coefs_linear(logger, modneutral)
        set_coefs_linear(logger, modlow)
        set_coefs_linear(logger, modhigh)

        # Run Quasoare
        t0 = time.time()
        modref.run()
        dt_quas = time.time() - t0

        t0 = time.time()
        modneutral.run()
        dt_quas_neutral = time.time() - t0

        # Print one simulation to make sure it makes sense
        if irepeat == 0:
            logger.info("Plotting first simul")
            nval = min(2000, len(pin))
            kk = np.arange(-nval, 0)
            nsims = nres_per_reach + 2
            df = np.empty((nval, nreaches * nsims))
            cols = []
            nrp = nres_per_reach
            for ireach in range(nreaches):
                # Data
                df[:, ireach * nsims] = pin[kk, ireach]
                df[:, ireach * nsims + 1] = ein[kk, ireach]
                # Sim - rr prod
                df[:, ireach * nsims + 2] = -modref.fluxes[kk, nrp * ireach, 5]
                # Sim - rr route
                df[:, ireach * nsims + 3] = -modref.fluxes[kk, nrp * ireach + 1, 5]
                # Sim - ro route
                df[:, ireach * nsims + 4] = -modref.fluxes[kk, nrp * ireach + 2, 5]
                # Sim - sum
                df[:, ireach * nsims + 5] = -modref.fluxes[kk, nrp * ireach + 3, 5]
                r = f"_{ireach:03d}"
                cols.extend([f"P{r}", f"PE{r}", f"RR_PROD{r}",
                             f"RR_ROUTE{r}", f"RO_ROUTE{r}",
                             f"AGG{r}"])

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
                    continue

                r = f"_{ireach:03d}"
                df.loc[:, f"RO_ROUTE{r}"].plot(ax=ax, label="RO-Route")
                df.loc[:, f"AGG{r}"].plot(ax=ax, label="Agg")

                tax = ax.twinx()
                df.loc[:, f"RR_ROUTE{r}"].plot(ax=tax, color="0.5")
                ax.plot([], [], color="0.5", label="RR-Route")

                ax.set_yscale("function",
                              functions=(forward_plot_scale,
                                         backward_plot_scale))
                ax.set_title(f"Reach {ireach}")
                ax.legend(loc=2)

            fig.savefig(fs.parent / f"{fs.stem}.png")

        # --------------------------------------------------
        logger.info("Run numerical difference - grad model")
        # .. run diff for the capacity of reach reservoir - order 1
        t0 = time.time()
        modneutral.run()
        for res in np.arange(nres):
            # Run high model
            modhigh.gammas[res] = modref.gammas[res] + numdiff_eps
            modhigh.run()

            # Compute flux gradient
            dfluxes_quas_diff1[:, res, :, :] = (modhigh.fluxes - modneutral.fluxes) / numdiff_eps

            # Reset
            modhigh.gammas[res] = modref.gammas[res]

        dt_quas_diff1 = time.time() - t0

        # --------------------------------------------------
        # .. run diff for the capacity of reach reservoir - order 2
        t0 = time.time()
        for res in np.arange(nres):
            # Run low model
            modlow.gammas[res] = modref.gammas[res] - numdiff_eps
            modlow.run()
            # Run high model
            modhigh.gammas[res] = modref.gammas[res] + numdiff_eps
            modhigh.run()

            # Compute flux gradient
            dfluxes_quas_diff2[:, res, :, :] = (modhigh.fluxes - modlow.fluxes) / 2 / numdiff_eps

            # Reset
            modlow.gammas[res] = modref.gammas[res]
            modhigh.gammas[res] = modref.gammas[res]

        dt_quas_diff2 = time.time() - t0

        # --------------------------------------------------
        # Run numerical difference - bench model
        # .. run diff for the capacity of reach reservoir - order 1
        out = x_rr_prod, x_rr_route, x_ro_route, \
            fx_rr_prod, fx_rr_route, fx_ro_route
        out_p = x_rr_prod, x_rr_route, x_ro_route, \
            fx_rr_prod_p, fx_rr_route_p, fx_ro_route_p

        t0 = time.time()
        benchmarks.gradbench_semidist(MODEL_TIMESTEP,
                                      pin, ein, x0, areas,
                                      grr_prod,
                                      grr_route,
                                      gro_route,
                                      demrr_route, out)
        for res in np.arange(nres):
            # Set params
            ireach = res // nres_per_reach
            ires_in_reach = res % nres_per_reach
            if ires_in_reach == 0:
                grr_prod_p[ireach] = grr_prod[ireach] + numdiff_eps
            elif ires_in_reach == 1:
                grr_route_p[ireach] = grr_route[ireach] + numdiff_eps
            elif ires_in_reach == 3:
                gro_route_p[ireach] = gro_route[ireach] + numdiff_eps

            # Run high model
            benchmarks.gradbench_semidist(MODEL_TIMESTEP,
                                          pin, ein, x0, areas,
                                          grr_prod_p,
                                          grr_route_p,
                                          gro_route_p,
                                          demrr_route, out_p)
            # Compute flux gradient
            # .. rr prod
            fx_p, fx = out_p[3], out[3]
            kk = np.arange(0, nres, nres_per_reach)
            dfluxes_bench_diff1[:, res, kk, 5] = -(fx_p - fx) / numdiff_eps

            # .. rr route
            fx_p, fx = out_p[4], out[4]
            kk = np.arange(1, nres, nres_per_reach)
            dfluxes_bench_diff1[:, res, kk, 5] = -(fx_p - fx) / numdiff_eps
            # .. ro route
            fx_p, fx = out_p[5], out[5]
            kk = np.arange(2, nres, nres_per_reach)
            dfluxes_bench_diff1[:, res, kk, 5] = -(fx_p - fx) / numdiff_eps

            # Reset
            if ires_in_reach == 0:
                grr_prod_p[ireach] = grr_prod[ireach]
            elif ires_in_reach == 1:
                grr_route_p[ireach] = grr_route[ireach]
            elif ires_in_reach == 3:
                gro_route_p[ireach] = gro_route[ireach]

        dt_bench_diff1 = time.time() - t0

        # --------------------------------------------------
        # Run numerical difference - bench model
        # .. run diff for the capacity of reach reservoir - order 2
        out_m = x_rr_prod, x_rr_route, x_ro_route, \
            fx_rr_prod_m, fx_rr_route_m, fx_ro_route_m

        t0 = time.time()
        benchmarks.gradbench_semidist(MODEL_TIMESTEP,
                                      pin, ein, x0, areas,
                                      grr_prod,
                                      grr_route,
                                      gro_route,
                                      demrr_route, out)
        for res in np.arange(nres):
            # Set params
            ireach = res // nres_per_reach
            ires_in_reach = res % nres_per_reach
            if ires_in_reach == 0:
                grr_prod_p[ireach] = grr_prod[ireach] + numdiff_eps
                grr_prod_m[ireach] = grr_prod[ireach] - numdiff_eps
            elif ires_in_reach == 1:
                grr_route_p[ireach] = grr_route[ireach] + numdiff_eps
                grr_route_m[ireach] = grr_route[ireach] - numdiff_eps
            elif ires_in_reach == 3:
                gro_route_p[ireach] = gro_route[ireach] + numdiff_eps
                gro_route_m[ireach] = gro_route[ireach] - numdiff_eps

            # Run high model
            benchmarks.gradbench_semidist(MODEL_TIMESTEP,
                                          pin, ein, x0, areas,
                                          grr_prod_p,
                                          grr_route_p,
                                          gro_route_p,
                                          demrr_route, out_p)
            # Run low model
            benchmarks.gradbench_semidist(MODEL_TIMESTEP,
                                          pin, ein, x0, areas,
                                          grr_prod_m,
                                          grr_route_m,
                                          gro_route_m,
                                          demrr_route, out_m)

            # Compute flux gradient
            # .. rr prod
            fx_p, fx_m = out_p[3], out_m[3]
            kk = np.arange(0, nres, nres_per_reach)
            dfluxes_bench_diff2[:, res, kk, 5] = -(fx_p - fx_m) / numdiff_eps / 2
            # .. rr route
            fx_p, fx_m = out_p[4], out_m[4]
            kk = np.arange(1, nres, nres_per_reach)
            dfluxes_bench_diff2[:, res, kk, 5] = -(fx_p - fx_m) / numdiff_eps / 2
            # .. ro route
            fx_p, fx_m = out_p[5], out_m[5]
            kk = np.arange(2, nres, nres_per_reach)
            dfluxes_bench_diff2[:, res, kk, 5] = -(fx_p - fx_m) / numdiff_eps / 2

            # Reset
            if ires_in_reach == 0:
                grr_prod_p[ireach] = grr_prod[ireach]
                grr_prod_m[ireach] = grr_prod[ireach]
            elif ires_in_reach == 1:
                grr_route_p[ireach] = grr_route[ireach]
                grr_route_m[ireach] = grr_route[ireach]
            elif ires_in_reach == 3:
                gro_route_p[ireach] = gro_route[ireach]
                gro_route_m[ireach] = gro_route[ireach]

        dt_bench_diff2 = time.time() - t0

        # --------------------------------------------------
        logger.info("Postprocess data")
        # Convert the routing fluxes in m3/s to mm/d
        # + convert grad against capacity to grad against capacity
        # scaling
        area, nrp, gams = modref.area, modref.nres_per_reach, modref.gammas
        fluxes_quas_c, dfluxes_quas_c = convert_fluxes(area, nrp, gams,
                                                       modref.fluxes,
                                                       modref.dfluxes)
        fluxes_bench_c, dfluxes_bench_c = convert_fluxes(area, nrp, gams,
                                                         fluxes_bench,
                                                         dfluxes_bench)
        _, dfluxes_quas_diff1_c = convert_fluxes(area, nrp, gams,
                                            fluxes_bench,
                                            dfluxes_quas_diff1)

        _, dfluxes_quas_diff2_c = convert_fluxes(area, nrp, gams,
                                            fluxes_bench,
                                            dfluxes_quas_diff2)

        _, dfluxes_bench_diff1_c = convert_fluxes(area, nrp, gams,
                                            fluxes_bench,
                                            dfluxes_bench_diff1)

        _, dfluxes_bench_diff2_c = convert_fluxes(area, nrp, gams,
                                            fluxes_bench,
                                            dfluxes_bench_diff2)

        # Compute perf
        results.loc[irepeat, "runtime_bench"] = dt_bench
        results.loc[irepeat, "runtime_quasoare_diff1"] = dt_quas_diff1
        results.loc[irepeat, "runtime_quasoare_diff2"] = dt_quas_diff2
        results.loc[irepeat, "runtime_bench_diff1"] = dt_bench_diff1
        results.loc[irepeat, "runtime_bench_diff2"] = dt_bench_diff2
        results.loc[irepeat, "runtime_quasoare"] = dt_quas
        results.loc[irepeat, "runtime_quasoare_neutral"] = dt_quas_neutral

        nan = "keep" if debug else "ignore"
        emax, coefdet = compute_flux_error(fluxes_bench_c, fluxes_quas_c, nan=nan)
        results.loc[irepeat, "fx_errmax_quasoare"] = emax
        results.loc[irepeat, "fx_coefdet_quasoare"] = coefdet

        emax, coefdet = compute_dflux_error(dfluxes_bench_c, dfluxes_quas_c, nan=nan)
        results.loc[irepeat, "dfx_errmax_quasoare"] = emax
        results.loc[irepeat, "dfx_coefdet_quasoare"] = coefdet

        emax, coefdet = compute_dflux_error(dfluxes_bench_c, dfluxes_quas_diff1_c)
        results.loc[irepeat, "dfx_errmax_quasoare_diff1"] = emax
        results.loc[irepeat, "dfx_coefdet_quasoare_diff1"] = coefdet

        emax, coefdet = compute_dflux_error(dfluxes_bench_c, dfluxes_quas_diff2_c)
        results.loc[irepeat, "dfx_errmax_quasoare_diff2"] = emax
        results.loc[irepeat, "dfx_coefdet_quasoare_diff2"] = coefdet

        emax, coefdet = compute_dflux_error(dfluxes_bench_c, dfluxes_bench_diff1_c)
        results.loc[irepeat, "dfx_errmax_bench_diff1"] = emax
        results.loc[irepeat, "dfx_coefdet_bench_diff1"] = coefdet

        emax, coefdet = compute_dflux_error(dfluxes_bench_c, dfluxes_bench_diff2_c)
        results.loc[irepeat, "dfx_errmax_bench_diff2"] = emax
        results.loc[irepeat, "dfx_coefdet_bench_diff2"] = coefdet

    results.index.name = "REPEAT"
    fres =  fperf / f"performance_v{version}_TASK{taskid}.csv"
    csv.write_csv(results, fres, "Performance",
                  script_paths.source_file,
                  write_index=True,
                  lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark gradient estimation against analytical solution and numerical difference",
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
        task.nreaches = 3

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
