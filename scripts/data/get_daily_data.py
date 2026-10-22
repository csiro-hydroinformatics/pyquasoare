#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-07-17 09:19:18.496670
## Comment : Collect daily data
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

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils, hyruns

from netCDF4 import Dataset, num2date

def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot.parent.parent / "Data" / "characterisation_paper"
    if not fdata.exists():
        raise ValueError(f"{fdata} does not exist")

    fout = froot / "data" / "daily"

    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fout"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fout)
    fout.mkdir(exist_ok=True, parents=True)
    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    logger = iutils.get_logger(basename, console=True)
    logger.log_dict(config._asdict(), "Configuration")
    logger.info("", nret=1)
    return logger


def get_data(config, script_paths, logger):
    fs = script_paths.fout.parent / "sites.csv"
    stations, _ = csv.read_csv(fs, index_col="STATIONID",
                               dtype={"STATIONID": str})
    data = namedtuple("Data", ["stations"])(stations)
    return data


def process(config, script_paths, logger, data):
    stations = data.stations
    nstations = len(stations)
    logger.info(f"Start processing", nret=1)

    fnc = script_paths.fdata / f"combined_daily_v{config.version}.nc"
    with Dataset(fnc, "r") as nc:
        # Select netcdf data
        vsid = "station_id"
        stationids = nc[vsid][:]

        cns = nc["station_metadata.column_name.numerical"][:]
        iarea = list(cns).index("CATCHMENTAREA_VALID")
        areas = nc["station_metadata.numerical"][:, iarea]

        vtime = "station_data_index"
        units = nc[vtime].units
        time = pd.to_datetime(num2date(nc[vtime][:], units,
                                       only_use_cftime_datetimes=False))

        cnum = list(nc["catchment_data_daily.column_name.numerical"][:])

        cq = "STREAMFLOW_DAILYMEAN_9AM"
        cp = "PRECIPITATION_AWAP_MEAN_9AM"
        ce = "ETMORTONWET_SILO_MEAN_9AM"
        icol = [cnum.index(c) for c in [cp, ce, cq]]
        units = [nc["catchment_data_daily.column_units.numerical"][i]
                 for i in icol]
        columns = ["RAINFALL", "PET", "STREAMFLOW"]
        columns = [f"{cn}[{u}]" for cn, u in zip(columns, units)]
        vnum = "catchment_data_daily.numerical"

        for isite, (stationid, sinfo) in enumerate(stations.iterrows()):
            ctxt = f"{stationid} ({isite+1}/{nstations})"
            logger.info(f"Processing {ctxt}", nret=1)

            ista = np.where(stationids == stationid)[0][0]
            area = areas[ista]
            if abs(area - sinfo["CATCHMENTAREA[km2]"]) > 1:
                raise ValueError("Problem with area")

            df = pd.DataFrame(nc[vnum][ista, :, icol],
                              index=time, columns=columns)
            idx = df.loc[:, columns[:2]].notnull().all(axis=1)
            df = df.loc[idx]
            df.loc[:, "STREAMFLOW[mm.day-1]"] = df.loc[:, columns[-1]] / area *  86.4

            # Reorder columns
            cols = list(df.columns)
            df = df.loc[:, cols[:2] + [cols[-1]]]

            logger.info(f"Nval = {len(df)}")
            logger.info("Mean annual:", nret=1)
            logger.info((df.mean() * 365.25).astype(int))
            logger.info("Valid years:", nret=1)
            logger.info((df.notnull().sum() / 365.25).astype(int))

            fd = script_paths.fout / f"hydrodata_{stationid}.csv"
            csv.write_csv(df, fd, f"Daily hydroclimate for station {stationid}",
                          script_paths.source_file, write_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect daily data",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version",
                        help="Version number",
                        type=str, required=True)
    args = parser.parse_args()

    # Config
    version = args.version
    Config = namedtuple("Config", ["version"])
    config = Config(version)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
