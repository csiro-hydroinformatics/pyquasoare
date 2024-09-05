#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2024-09-05 11:30:56.386150
## Comment : Compute summary stats from catchment data
##
## ------------------------------


import sys, os, re, json, math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

from hydrodiy.io import csv, iutils

import data_utils

#----------------------------------------------------------------------
# @Config
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fout = froot / "outputs"
fout.mkdir(exist_ok=True)

#----------------------------------------------------------------------
# @Logging
#----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# @Get data
#----------------------------------------------------------------------
sites = data_utils.get_sites()

#----------------------------------------------------------------------
# @Process
#----------------------------------------------------------------------
fmt = lambda x: f"{x:0,.0f} km2"
dd = {\
    "name": sites.NAME.str.replace(".* At ", "", regex=True), \
    "down": sites.NAME + "\nSite ID " + sites.index + ", " \
                +sites.loc[:, "CATCHMENTAREA[km2]"].apply(fmt), \
    "up": sites.UPSTREAM_NAME + "\nSite ID " + sites.UPSTREAM_STATIONID \
            + ", " +sites.loc[:, "UPSTREAM_CATCHMENTAREA[km2]"].apply(fmt)
}

stats = pd.DataFrame(dd)
stats.loc[:, "2022_Rain"] = 0
stats.loc[:, "2022_PET"] = 0
stats.loc[:, "2022_Peak"] = 0

for siteid, info in sites.iterrows():
    daily = data_utils.get_data(siteid, "daily")
    hourly = data_utils.get_data(siteid, "hourly")

    rain = daily.loc["2022", "RAINFALL[mm/day]"].sum()
    stats.loc[siteid, "2022_Rain"] = int(rain)

    pet = daily.loc["2022", "PET[mm/day]"].sum()
    stats.loc[siteid, "2022_PET"] = int(pet)

    peak = hourly.loc["2022", "STREAMFLOW_DOWN[m3/sec]"].max()
    stats.loc[siteid, "2022_Peak"] = int(peak)

fs = fout / "sites_stats.csv"
csv.write_csv(stats, fs, "Sites stats", \
        source_file, compress=False, write_index=True)

LOGGER.completed()

