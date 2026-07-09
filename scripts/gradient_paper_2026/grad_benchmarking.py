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


import importlib.util
FTESTS = Path(__file__).parent.parent.parent / "tests"
spec = importlib.util.spec_from_file_location("data_reader", FTESTS / "data_reader.py")
data_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_reader)

np.random.seed(5446)

def get_options():
    timestep = "daily"
    duration_base = 5
    duration_total = 100

    stationids = data_reader.SITEIDS
    nreaches = [1, 2, 5, 10, 20, 50, 100]

    nrepeat = 100
    nbatch = 5
    batches = np.arange(nbatch)

    gamma_means = [10, 100, 1000]
    gamma_std_prop = 0.1

    opm = hyruns.OptionManager(duration_base=duration_base,
                               duration_total=duration_total,
                               gamma_std_prop=gamma_std_prop,
                               timestep=timestep,
                               nrepeat=nrepeat,
                               nbatch=nbatch)

    opm.from_cartesian_product(batch=batches,
                               stationid=stationids,
                               nreaches=nreaches,
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
    timestep = config.options.context["timestep"]

    data = data_reader.get_data(stationid, timestep)
    nval = config.options.context["duration_base"] * 365
    ntot = config.options.context["duration_total"] * 365
    nreps = ntot // nval

    hydroclim = np.tile(data.iloc[-nval:, :2], (nreps, 1))
    hydroclim = pd.DataFrame(hydroclim, columns=["P", "PET"])

    data = namedtuple("Data", ["hydroclim"])(hydroclim)

    return data


def process(config, script_paths, logger, data):
    # Define job
    batch = config.task.batch
    nreaches = config.task.nreaches
    gamma_mean = config.task.gamma_mean
    gamma_std_prop = config.options.context["gamma_std_prop"]
    nbatch = config.options.context["nbatch"]
    nrepeat = config.options.context["nrepeat"]
    ibatches = np.array_split(np.arange(nrepeat), nbatch)[batch]

    # Get data
    hydroclim = data.hydroclim

    # Loop through random repeat
    for ibatch in ibatches:
        nres = 3 * nreaches
        gammas = gamma_mean * (1 + np.random.normal(scale=gamma_std_prop, size=nres))


    import pdb; pdb.set_trace()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark gradient estimation against analytical solution and numerical difference",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version",
                        help="Version number",
                        type=str, required=True)
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
