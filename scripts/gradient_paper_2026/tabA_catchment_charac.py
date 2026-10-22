#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-09-22 Tue 09:35 AM
## Comment : Generate table of catchment characteristics
##
## ------------------------------

import sys
import re
import json
import math
from itertools import product as prod
import argparse
from datetime import datetime
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

from gradbench import data_reader

def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fimg = froot / "images" / "manuscript"
    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fimg"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fimg)
    fimg.mkdir(exist_ok=True, parents=True)

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    logger = iutils.get_logger(basename, console=True)
    logger.log_dict(config._asdict(), "Configuration")
    logger.info("", nret=1)
    return logger


def get_data(config, script_paths, logger):
    stations = data_reader.get_stations()
    hclim = {}
    for stationid in stations.index:
        df = data_reader.get_data(stationid, timestep, nval=None)
        hclim[stationid] = df
    data = namedtuple("Data", ["stations", "hclim"])(stations, hclim)
    return data


def process(config, script_paths, logger, data):
    stations = data.stations
    tab = []
    for stationid, sinfo in stations.iterrows():
        df = data.hclim[stationid]
        me = df.mean() * 365.25
        mx = df.max()
        dd = {
            "Station ID": stationid,
            "Name": re.sub(" At ", " at ", sinfo.NAME),
            "Catchment area [km2]": int(sinfo["CATCHMENTAREA[km2]"]),
            "Mean annual Rainfall [mm/yr]": int(me.loc["RAINFALL[mm_day-1]"]),
            "Mean annual PET [mm/yr]": int(me.loc["PET[mm_day-1]"]),
            "Mean annual Streamflow [mm/yr]": int(me.loc["STREAMFLOW[mm_day-1]"]),
            "Maximum daily rainfall [mm/day]": int(mx.loc["RAINFALL[mm_day-1]"]),
            }
        tab.append(dd)

    tab = pd.DataFrame(tab)
    fc = script_paths.fimg / "tabA_catchment_characteristics.csv"
    csv.write_csv(tab, fc, "Catchment charac",
                  script_paths.source_file, compress=False,
                  lineterminator="\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table of catchment charac",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--debug",
                        help="Debug mode",
                        action="store_true", default=False)
    args = parser.parse_args()

    timestep = "daily"

    # Config
    Config = namedtuple("Config", ["debug", "timestep"])
    config = Config(args.debug, timestep)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
