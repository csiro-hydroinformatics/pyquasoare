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

def get_options(debug=False):
    timestep = "daily"
    duration_base = 100
    duration_repeat = 5

    stationids = data_reader.SITEIDS
    nreaches = [1, 5, 10, 20, 50, 100]
    grad_methods = ["DIFF1", "DIFF2", "QUASOARE"]
    nalphas = [10, 50, 100, 200, 500, 1000]

    nrepeat = 100
    nbatch = 5
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
                               gamma_rr_mean=gamma_rr_means,
                               gamma_ro_mean=gamma_ro_means,
                               grad_method=grad_methods)
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

    return mod


def allocate_model(config, hydroclim, area, mod):
    nval = len(hydroclim)
    mod.allocate(nval, timestep=MODEL_TIMESTEP)

    # Set 0 flow inputs for first reach
    Qstars = np.zeros((nval, 2))
    mod.set_Qstars(3, Qstars)

    # Transform inputs
    tP = forwardP(hydroclim[:, 0])
    rhoP = pd.Series(tP).autocorr()
    alphaP = math.sqrt(1 - rhoP**2) * tP.std() / 10

    tE = forwardE(hydroclim[:, 1])
    rhoE = pd.Series(tE).autocorr()
    alphaE = math.sqrt(1 - rhoE**2) * tE.std() / 10

    # Set inputs
    nreaches = config.task.nreaches
    nres_per_reach = 4
    for ireach in range(nreaches):
        # Reach input noise
        # .. rainfall
        nu = np.random.normal(scale=rhoP, size=nval)
        errP = armodels.armodel_sim(alphaP, nu)
        Qstars[:, 0] = backwardP(tP + errP)

        # .. PET
        nu = np.random.normal(scale=rhoE, size=nval)
        errE = 1 + armodels.armodel_sim(alphaE, nu)
        Qstars[:, 1] = backwardE(tE + errE)

        ibase = ireach * nres_per_reach
        mod.set_Qstars(ibase, Qstars)


def process(config, script_paths, logger, data):
    # Setup up model
    area = data.area
    mod = configure_model(config, area)

    # Allocate model
    context = config.options.context
    dbase = context["duration_base"]
    drep = context["duration_repeat"]
    n = dbase * 365
    hclim = np.tile(data.hydroclim.iloc[-n:], (drep, 1))
    allocate_model(config, hclim, area, mod)

    import pdb; pdb.set_trace()


    # Define job
    batch = config.task.batch
    nreaches = config.task.nreaches
    gamma_rr_mean = config.task.gamma_rr_mean
    gamma_rr_std = config.options.context["gamma_rr_std"]
    gamma_ro_mean = config.task.gamma_rr_mean
    gamma_ro_std = config.options.context["gamma_ro_std"]
    nbatch = config.options.context["nbatch"]
    nrepeat = config.options.context["nrepeat"]
    ibatches = np.array_split(np.arange(nrepeat), nbatch)[batch]
    grad_method = config.task.grad_method
    nalphas = config.task.nalphas

    for irepeat in range(nrepeat):
        # Set reservoirs
        gammas = np.zeros(nres)
        # .. RR prod reservoir
        s = (1 + np.random.normal(scale=gamma_rr_std, size=nreaches))
        grr_prod = gamma_rr_mean * s
        gammas[::4] = grr_prod

        # .. RR route reservoir
        s = (1 + np.random.normal(scale=gamma_rr_std, size=nreaches))
        grr_route = gamma_rr_mean / 10. * s
        gammas[1::4] = grr_route

        # .. RO reservoir
        factor = Pmean * area * np.arange(1, nreaches + 1) / 86.4 * gamma_ro_mean
        s = (1 + np.random.normal(scale=gamma_ro_std, size=nreaches))
        gro_route = factor * s
        gammas[2::4] = gro_route

        # .. transluent capacity
        gammas[3::4] = 1.

        # Run benchmark
        t0 = time.time()
        out = benchmarks.gradbench_semidist_tangent(timestep, pin, ein, x0, areas,
                                                    grr_prod,
                                                    dgammas_rr_prod,
                                                    grr_route,
                                                    dgammas_rr_route,
                                                    gro_route,
                                                    dgammas_ro_route,
                                                    drr_route,
                                                    ddemands_rr_route)
        dtb = time.time() - t0
        x_rr_prod, dx_rr_prod, \
            x_rr_route, dx_rr_route, \
            x_ro_route, dx_ro_route, \
            fx_rr_prod, dfx_rr_prod, \
            fx_rr_route, dfx_rr_route, \
            fx_ro_route, dfx_ro_route = out


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
