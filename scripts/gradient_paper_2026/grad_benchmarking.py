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

from pyquasoare import gradients
from pyquasoare.benchmarks import benchmarks

import importlib.util
FTESTS = Path(__file__).parent.parent.parent / "tests"
spec = importlib.util.spec_from_file_location("data_reader", FTESTS / "data_reader.py")
data_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_reader)

np.random.seed(5446)


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
    duration_base = 5
    duration_total = 100

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

    opm = hyruns.OptionManager(duration_base=duration_base,
                               duration_total=duration_total,
                               gamma_rr_std=gamma_rr_std,
                               gamma_ro_std=gamma_ro_std,
                               timestep=timestep,
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
    area = 0.

    nval = config.options.context["duration_base"] * 365
    ntot = config.options.context["duration_total"] * 365
    nreps = ntot // nval

    timestep = config.options.context["timestep"]
    data = data_reader.get_data(stationid, timestep)
    hydroclim = np.tile(data.iloc[-nval:, :2], (nreps, 1))
    hydroclim = pd.DataFrame(hydroclim, columns=["P", "PET"])

    data = namedtuple("Data", ["hydroclim", "area"])(hydroclim, area)

    return data


def process(config, script_paths, logger, data):
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

    # Get data
    hydroclim = data.hydroclim

    # Configure model
    nres_per_reach = 4
    nres = nres_per_reach * nreaches
    mod = gradients.MultiReservoirModel(nres, errors="raise")
    mod.alphas = nalphas
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

    # Loop through random repeat
    for ibatch in ibatches:
        # RR reservoir capacity
        gammas = gamma_rr_mean * (1 + np.random.normal(scale=gamma_rr_std, size=nres))

        # .. transluent capacity
        gammas[::4] = 1.

        # .. RO reservoir capacity
        # .. estimates qmean

        #gammas[::3] =


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
