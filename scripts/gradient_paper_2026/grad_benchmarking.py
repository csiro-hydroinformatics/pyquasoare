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

def forwardP(x):
    return np.sqrt(np.sqrt(x))

def backwardP(y):
    return np.maximum(y, 0.)**4

def forwardE(x):
    return x

def backwardE(y):
    return np.maximum(y, 0.)

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

    gamma_rr_means = [10, 100, 1000]
    gamma_rr_std = 0.1
    gamma_ro_means = [0.1, 1., 10.]
    gamma_ro_std = 0.1
    demand_mean = -0.5
    demand_std = 0.2

    opm = hyruns.OptionManager(duration_base=duration_base,
                               duration_repeat=duration_repeat,
                               gamma_rr_std=gamma_rr_std,
                               gamma_ro_std=gamma_ro_std,
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
                               gamma_rr_mean=gamma_rr_means,
                               gamma_ro_mean=gamma_ro_means)
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
    mod = gradients.MultiReservoirModel(nres, errors="raise")
    mod.alphas = config.task.nalphas
    mod.area = area

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

        # .. links ...
        # RR prod -> RR routing
        mod.add_link_outflow(ibase, ibase + 1)

        # RR route -> transluent with conversion
        conv = area * MODEL_TIMESTEP / 86.4
        mod.add_link_outflow(ibase + 1, ibase + 3, conv)

        # Transluent upstream -> RO route
        if ireach > 0:
            mod.add_link_outflow(ibase - 1, ibase + 2)

    mod.nreaches = nreaches
    mod.nres_per_reach = nres_per_reach

    return mod


def allocate_model(mod, hydroclim, grad=True):
    nval = len(hydroclim)

    # .. gradients against all capacities
    if grad:
        vspecs = [[res, 0] for res in range(mod.nreservoirs)]
    else:
        vspecs = None
    mod.allocate(nval, timestep=MODEL_TIMESTEP,
                 variables_specs=vspecs)

    mod.Pmean = hydroclim[:, 0].mean()

    # Set 0 flow inputs for first reach
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
    gamma_rr_mean = config.task.gamma_rr_mean
    gamma_rr_std = config.options.context["gamma_rr_std"]
    gamma_ro_mean = config.task.gamma_rr_mean
    gamma_ro_std = config.options.context["gamma_ro_std"]

    gammas = np.zeros(nres)
    # .. RR prod reservoir
    s = (1 + np.random.normal(scale=gamma_rr_std, size=nreaches))
    grr_prod = gamma_rr_mean * s
    gammas[::nrp] = grr_prod

    # .. RR route reservoir
    s = (1 + np.random.normal(scale=gamma_rr_std, size=nreaches))
    grr_route = gamma_rr_mean / 10. * s
    gammas[1::nrp] = grr_route

    # .. RO reservoir
    factor = mod.Pmean * mod.area * np.arange(1, nreaches + 1) / 86.4 * gamma_ro_mean
    s = (1 + np.random.normal(scale=gamma_ro_std, size=nreaches))
    gro_route = factor * s
    gammas[2::nrp] = gro_route

    # .. transluent capacity
    gammas[3::nrp] = 1.

    mod.gammas = gammas

    # Demand moments
    demand_mean = config.options.context["demand_mean"]
    demand_std = config.options.context["demand_std"]
    dem = np.random.normal(size=nreaches, loc=demand_mean, scale=demand_std)
    mod.flux_scalings[1::nrp, 1] = dem

    return grr_prod, grr_route, gro_route, dem


def process(config, script_paths, logger, data):
    # Setup up model
    modref = configure_model(config, data.area)
    # .. difference models
    modlow = configure_model(config, data.area)
    modhigh = configure_model(config, data.area)

    # Allocate model with extended hydroclim data
    debug = config.debug
    context = config.options.context
    dbase = 10 if debug else context["duration_base"]
    drep = 1 if debug else context["duration_repeat"]
    n = dbase * 365
    hclim = np.tile(data.hydroclim.iloc[-n:], (drep, 1))
    allocate_model(modref, hclim)
    allocate_model(modlow, hclim, False)
    allocate_model(modhigh, hclim, False)

    # Define job
    batch = config.task.batch
    nbatch = config.options.context["nbatch"]
    nrepeat = config.options.context["nrepeat"]
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
    x_rr_prod = np.empty((nrepeat, nval, nreaches))
    dx_rr_prod = np.empty((nrepeat, nres, nval, nreaches))
    x_rr_route = np.empty((nrepeat, nval, nreaches))
    dx_rr_route = np.empty((nrepeat, nres, nval, nreaches))
    x_ro_route = np.empty((nrepeat, nval, nreaches))
    dx_ro_route = np.empty((nrepeat, nres, nval, nreaches))

    fx_rr_prod = np.empty((nrepeat, nval, nreaches))
    dfx_rr_prod = np.empty((nrepeat, nres, nval, nreaches))
    fx_rr_route = np.empty((nrepeat, nval, nreaches))
    dfx_rr_route = np.empty((nrepeat, nres, nval, nreaches))
    fx_ro_route = np.empty((nrepeat, nval, nreaches))
    dfx_ro_route = np.empty((nrepeat, nres, nval, nreaches))

    # initialise numerical difference data
    numdiff_eps = config.task.numdiff_eps
    dfluxes_diff = modref.dfluxes.copy()

    # repeat experiment
    for irepeat in range(nrepeat):
        grr_prod, grr_route, gro_route, demrr_route = \
                set_parameters(config, modref)
        modlow.gammas = modref.gammas
        modlow.flux_scalings = modref.flux_scalings

        modhigh.gammas = modref.gammas
        modhigh.flux_scalings = modref.flux_scalings

        # Run benchmark
        pin = modref.Qstars[:, ::modref.nres_per_reach, 0]
        ein = modref.Qstars[:, ::modref.nres_per_reach, 1]

        t0 = time.time()
        # .. run tangent code for the capacity of reach reservoir
        for res in range(nres):
            out = x_rr_prod[irepeat], dx_rr_prod[irepeat, res], \
                x_rr_route[irepeat], dx_rr_route[irepeat, res], \
                x_ro_route[irepeat], dx_ro_route[irepeat, res], \
                fx_rr_prod[irepeat], dfx_rr_prod[irepeat, res], \
                fx_rr_route[irepeat], dfx_rr_route[irepeat, res], \
                fx_ro_route[irepeat], dfx_ro_route[irepeat, res], \

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

        # Run Quasoare
        t0 = time.time()
        modref.run()
        dt_quas = time.time() - t0

        # Run numerical difference
        # .. run diff for the capacity of reach reservoir
        t0 = time.time()
        for res in range(nres):
            # Run low model
            modlow.gammas[res] = modref.gammas[res] - numdiff_eps
            modlow.run()

            modhigh.gammas[res] = modref.gammas[res] + numdiff_eps
            modhigh.run()

            # Compute flux gradient
            dfluxes_diff[:, res, :, :] = (modhigh.fluxes - modlow.fluxes) / 2 / numdiff_eps

            # Reset
            modlow.gammas[res] = modref.gammas[res]
            modhigh.gammas[res] = modref.gammas[res]

        dt_diff = time.time() - t0

        # Compute perf
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
