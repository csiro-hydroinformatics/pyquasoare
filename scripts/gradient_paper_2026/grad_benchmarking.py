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

#import warnings
#warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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
        lambda x: 1 - np.minimum(x, 1)**2,
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
    duration_base = 100
    duration_repeat = 5

    stationids = data_reader.SITEIDS
    nreaches = [1, 10, 100]
    nalphas = [10, 100, 1000]
    numdiff_eps = [1e-3, 1e-5, 1e-7]

    nrepeat = 100
    nbatch = 1
    batches = np.arange(nbatch)

    gamma_means = [
        (500, 50, 1),
        (50, 50, 1),
        (500, 5, 1),
        (500, 50, 10)
        ]
    gamma_std = 0.1
    demand_mean = -0.5
    demand_std = 0.2

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
    fl = script_paths.flogs / f"{basename}.log"
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
    stations = data_reader.get_stations()
    area = stations.loc[stationid, "CATCHMENTAREA[km2]"]

    nval = config.options.context["duration_base"] * 365
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

    return data

def configure_model(config, area):
    nreaches = config.task.nreaches
    nres_per_reach = 4
    nres = nres_per_reach * nreaches
    nalphas = 500 if config.debug else config.task.nalphas
    mod = gradients.MultiReservoirModel(nres, nalphas, errors="raise")
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

    return mod


def set_coefs_linear(mod):
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


def allocate_model(mod, hydroclim, grad=True):
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


def set_qstars(mod, hydroclim):
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


def set_parameters(config, mod):
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
    factor = mod.Pmean * mod.area * np.arange(1, nreaches + 1) / 86.4 * gamma_mean[2]
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


def compute_flux_error(bench, alt, transform=None):
    nreaches = bench.shape[1] // 4
    ires = (np.arange(nreaches)[:, None] + np.array([0, 1, 3])[None, :]).ravel()
    fa = alt[:, ires, 5]
    fb = bench[:, ires, 5]
    if transform is not None:
        fw = forward_asinh if transform=="asinh" else lambda x: x
        fb = fw(fb)
        fa = fw(fa)

    return np.abs(fa - fb)


def compute_dflux_error(dbench, dalt, transform=None):
    nreaches = dbench.shape[1] // 4
    ires = (np.arange(nreaches)[:, None] + np.array([0, 1, 3])[None, :]).ravel()

    # .. multiply err by gamma to get a dflux error with the dimension
    # of a flux
    dfb = dbench[:, ires, ires, 5]
    dfa = dalt[:, ires, ires, 5]
    if transform:
        fw = forward_asinh if transform=="asinh" else lambda x: x
        dfb = fw(dfb)
        dfa = fw(dfa)

    return np.abs(dfb - dfa)


def convert_fluxes(area, nres_per_reach, gammas,
                   fluxes, dfluxes):
    nres = dfluxes.shape[2]
    nreach = nres // nres_per_reach

    fluxes_converted = fluxes.copy()

    # Convert diff against capacity
    # to diff against capacity scaling
    g = gammas[None, None, :, None]
    dfluxes_converted = dfluxes.copy() * g

    # Convert m3/s to mm/day for the transluent reservoir
    for ireach in range(nreach):
        area_up = area * (ireach + 1)
        ires_transluent = ireach * nres_per_reach + 3
        factor = 86.4 * MODEL_TIMESTEP / area_up

        fluxes_converted[:, ires_transluent] = \
            factor * fluxes[:, ires_transluent]

        dfluxes_converted[:, :, ires_transluent] = \
            factor * dfluxes_converted[:, :, ires_transluent]

    return fluxes_converted, dfluxes_converted


def process(config, script_paths, logger, data):
    # Setup up model
    modref = configure_model(config, data.area)
    # .. difference models
    modneutral = configure_model(config, data.area)
    modlow = configure_model(config, data.area)
    modhigh = configure_model(config, data.area)

    # Allocate model with extended hydroclim data
    debug = config.debug
    context = config.options.context
    dbase = 20 if debug else context["duration_base"]
    drep = 1 if debug else context["duration_repeat"]
    n = dbase * 365
    hclim = np.tile(data.hydroclim.iloc[-n:], (drep, 1))
    allocate_model(modref, hclim)
    allocate_model(modneutral, hclim, False)
    allocate_model(modlow, hclim, False)
    allocate_model(modhigh, hclim, False)

    # Define job
    batch = config.task.batch
    nbatch = config.options.context["nbatch"]
    nrepeat = 10 if config.debug else config.options.context["nrepeat"]
    ibatches = np.array_split(np.arange(nrepeat), nbatch)[batch]

    # initialise gradient data
    nreaches = config.task.nreaches
    nres = modref.nreservoirs
    nres_per_reach = modref.nres_per_reach

    dgrr_prod = np.zeros((nres, nreaches))
    k = np.arange(nreaches) * nres_per_reach
    dgrr_prod[k, k] = 1

    dgrr_route = np.zeros((nres, nreaches))
    dgrr_route[k + 1, k] = 1

    dgro_route = np.zeros((nres, nreaches))
    dgro_route[k + 2, k] = 1

    ddemrr_route = np.zeros(nreaches)

    areas = np.ones(nreaches) * modref.area
    x0 = 0.5

    # initialise benchmark data
    nval = len(hclim)
    x_rr_prod = np.zeros((nval, nreaches))
    dx_rr_prod = np.zeros((nres, nval, nreaches))
    x_rr_route = np.zeros((nval, nreaches))
    dx_rr_route = np.zeros((nres, nval, nreaches))
    x_ro_route = np.zeros((nval, nreaches))
    dx_ro_route = np.zeros((nres, nval, nreaches))

    fx_rr_prod = np.zeros((nval, nreaches))
    dfx_rr_prod = np.zeros((nres, nval, nreaches))
    fx_rr_route = np.zeros((nval, nreaches))
    dfx_rr_route = np.zeros((nres, nval, nreaches))
    fx_ro_route = np.zeros((nval, nreaches))
    dfx_ro_route = np.zeros((nres, nval, nreaches))

    fluxes_bench = modref.fluxes.copy()
    fluxes_bench.fill(np.nan)
    dfluxes_bench = modref.dfluxes.copy()
    dfluxes_bench.fill(np.nan)

    # initialise numerical difference data
    numdiff_eps = 1e-2 if config.debug else config.task.numdiff_eps
    dfluxes_diff1 = modref.dfluxes.copy() * np.nan
    dfluxes_diff2 = modref.dfluxes.copy() * np.nan

    # initialise perf results
    mnames = ["bench", "diff1", "diff2", "quasoare"]
    cc = [f"runtime_{m}" for m in mnames]
    cc += [f"fx_errmax_quasoare"]
    cc += [f"dfx_errmax_{m}" for m in mnames[1:]]
    results = pd.DataFrame(0., index=np.arange(nrepeat),
                           columns=cc)

    # repeat experiment
    for irepeat in range(nrepeat):
        logger.info(f"Repeat #{irepeat + 1} / {nrepeat}")
        # Set random model inputs
        set_qstars(modref, hclim)
        modneutral.Qstars = modref.Qstars
        modlow.Qstars = modref.Qstars
        modhigh.Qstars = modref.Qstars

        grr_prod, grr_route, gro_route, demrr_route = \
                set_parameters(config, modref)

        modneutral.gammas = modref.gammas
        modneutral.flux_scalings = modref.flux_scalings

        modlow.gammas = modref.gammas
        modlow.flux_scalings = modref.flux_scalings

        modhigh.gammas = modref.gammas
        modhigh.flux_scalings = modref.flux_scalings

        # Run benchmark
        pin = modref.Qstars[:, ::modref.nres_per_reach, 0]
        # .. convert to positive ! -> classical convention in RR
        ein = -modref.Qstars[:, ::modref.nres_per_reach, 1]

        t0 = time.time()
        # .. run tangent code for the capacity of reach reservoir
        for res in range(nres):
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
                                                  ddemrr_route,
                                                  out)
        dt_bench = time.time() - t0

        # .. rearrange benchmark data to match model format
        for res in range(nres):
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

        # .. Modify alphas to match benchmark
        for res in range(nres):
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

        set_coefs_linear(modref)
        set_coefs_linear(modneutral)
        set_coefs_linear(modlow)
        set_coefs_linear(modhigh)

        # Run Quasoare
        t0 = time.time()
        modref.run()
        dt_quas = time.time() - t0

        # Run numerical difference
        # .. run diff for the capacity of reach reservoir - order 1
        t0 = time.time()
        modneutral.run()
        for res in range(nres):
            # Run low model
            modhigh.gammas[res] = modref.gammas[res] + numdiff_eps
            modhigh.run()

            # Compute flux gradient
            dfluxes_diff1[:, res, :, :] = (modhigh.fluxes - modneutral.fluxes) / numdiff_eps

            # Reset
            modhigh.gammas[res] = modref.gammas[res]

        dt_diff1 = time.time() - t0

        # .. run diff for the capacity of reach reservoir - order 2
        t0 = time.time()
        for res in range(nres):
            # Run low model
            modlow.gammas[res] = modref.gammas[res] - numdiff_eps
            modlow.run()

            modhigh.gammas[res] = modref.gammas[res] + numdiff_eps
            modhigh.run()

            # Compute flux gradient
            dfluxes_diff2[:, res, :, :] = (modhigh.fluxes - modlow.fluxes) / 2 / numdiff_eps

            # Reset
            modlow.gammas[res] = modref.gammas[res]
            modhigh.gammas[res] = modref.gammas[res]

        dt_diff2 = time.time() - t0

        # Convert the routing fluxes in m3/s to mm/d
        area, nrp, gams = modref.area, modref.nres_per_reach, modref.gammas
        fluxes_quas_c, dfluxes_quas_c = convert_fluxes(area, nrp, gams,
                                                       modref.fluxes,
                                                       modref.dfluxes)
        fluxes_bench_c, dfluxes_bench_c = convert_fluxes(area, nrp, gams,
                                                         fluxes_bench,
                                                         dfluxes_bench)
        _, dfluxes_diff1_c = convert_fluxes(area, nrp, gams,
                                            fluxes_bench,
                                            dfluxes_diff1)

        _, dfluxes_diff2_c = convert_fluxes(area, nrp, gams,
                                            fluxes_bench,
                                            dfluxes_diff2)

        # Compute perf
        results.loc[irepeat, "runtime_bench"] = dt_bench
        results.loc[irepeat, "runtime_diff1"] = dt_diff1
        results.loc[irepeat, "runtime_diff2"] = dt_diff2
        results.loc[irepeat, "runtime_quasoare"] = dt_quas

        fx_err = compute_flux_error(fluxes_bench_c, fluxes_quas_c)
        results.loc[irepeat, "fx_errmax_quasoare"] = fx_err.max()

        dfx_err = compute_dflux_error(dfluxes_bench_c, dfluxes_quas_c)
        results.loc[irepeat, "dfx_errmax_quasoare"] = dfx_err.max()

        if dfx_err.max() > 0.1:
            err = np.abs(dfluxes_bench_c - dfluxes_quas_c)

            import matplotlib.pyplot as plt
            db = dfluxes_bench_c[:, 0, 0, 5]
            dq = dfluxes_quas_c[:, 0, 0, 5]
            fig, ax = plt.subplots()
            ax.plot(db, label="bench")
            ax.plot(dq, label="quas")
            tax = ax.twinx()
            tax.plot(db - dq, color="0.5", lw=0.8)
            plt.show()
            import pdb; pdb.set_trace()

        dfx_err = compute_dflux_error(dfluxes_bench_c, dfluxes_diff1_c)
        results.loc[irepeat, "dfx_errmax_diff1"] = dfx_err.max()

        dfx_err = compute_dflux_error(dfluxes_bench_c, dfluxes_diff2_c)
        results.loc[irepeat, "dfx_errmax_diff2"] = dfx_err.max()

    import pdb; pdb.set_trace()


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
    config = Config(version, taskid, overwrite,
                    debug, opm, task)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
