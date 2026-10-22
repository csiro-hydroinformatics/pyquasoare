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
import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils, hyruns

from gradaverage import get_options

def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fout = froot / "outputs" / f"gradaverage_v{config.version}"
    flogs = froot / "logs" / source_file.stem

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fout", "flogs"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fout, flogs)
    flogs.mkdir(exist_ok=True, parents=True)

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    fl = script_paths.flogs / f"{basename}_TASK{config.taskid}.log"
    logger = iutils.get_logger(basename, flog=fl, console=True)
    return logger


def get_data(config, script_paths, logger):
    perf_files = list((script_paths.fout / "performance").glob("*.zip"))
    Data = namedtuple("Data", ["perf_files"])
    return Data(perf_files)

def process(config, script_paths, logger, data):
    opm = get_options()
    perf = []
    for f in data.perf_files:
        taskid = int(re.sub(".*_TASK", "", f.stem))
        task = opm.get_task(taskid)

        df, _ = csv.read_csv(f)
        df.loc[:, "TASKID"] = taskid
        for on, ovalue in task.options.items():
            if on == "gamma_mean":
                ovalue = "_".join([str(g) for g in ovalue])
            df.loc[:, f"TASK_OPT_{on}"] = ovalue

        perf.append(df)

    perf = pd.concat(perf, ignore_index=True)
    fp = script_paths.fout / f"performance_v{config.version}.csv"
    csv.write_csv(perf, fp, "Concatenated performance",
                  script_paths.source_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate results from benchmark gradient perf",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version",
                        help="Version number",
                        type=int, required=True)
    parser.add_argument("-t", "--taskid", help="JobID",
                        type=int, default=-1)
    parser.add_argument("-d", "--debug", help="Debug mode",
                        action="store_true", default=False)
    args = parser.parse_args()

    # Config
    version = args.version
    debug = args.debug
    taskid = 0 if debug else args.taskid

    Config = namedtuple("Config",
                        ["version", "taskid", "debug"])
    config = Config(version, taskid, debug)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
