#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-08-18 15:33:46.478450
## Comment : Figure presenting gradient benchmarking
##
## ------------------------------


import sys
import os
import re
import json
import math
import argparse
from datetime import datetime
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
from string import ascii_lowercase as letters

import matplotlib as mpl

import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils, violinplot

# Package to plot spatial data
from hydrodiy.io import csv, iutils

# Select backend
mpl.use("Agg")


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "outputs" / f"gradbench_v{config.version}"
    fimg = froot / "images" / "manuscript" / "gradient_benchmarking"
    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fimg"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fimg)
    fimg.mkdir(exist_ok=True, parents=True)

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    logger = iutils.get_logger(basename, console=True)
    logger.log_dict(config._asdict(), "Configuration")
    logger.info("", nret=1)
    return logger


def get_data(config, script_paths, logger):
    fp = script_paths.fdata / f"performance_v{config.version}.csv"
    perf, _ = csv.read_csv(fp)
    data = namedtuple("Data", ["perf"])(perf)
    return data


def process(config, script_paths, logger, data):

    logger.info(f"Start plotting", nret=1)

    # Get data
    perf = data.perf

    refs = {
            "nreaches": 5,
            "gamma_mean": "500_50_1000",
            "nalphas": 100
            }

    numdiff_log10eps = -5
    perf = perf.loc[perf.TASK_OPT_numdiff_log10eps == numdiff_log10eps]

    # Create figure
    for metric in ["errmax", "coefdet"]:
        plt.close("all")
        mosaic = [[f"fx/{vn}", f"dfx/{vn}"]
                  for vn in ["nreaches", "nalphas", "gamma_mean"]]
        fncols, fnrows = len(mosaic[0]), len(mosaic)
        awidth, aheight = config.awidth, config.aheight
        figsize = (awidth*fncols, aheight*fnrows)
        fig = plt.figure(constrained_layout=True,
                         figsize=figsize)

        gw = dict(height_ratios=[1]*fnrows, width_ratios=[1]*fncols)
        axs = fig.subplot_mosaic(mosaic, gridspec_kw=gw)

        for iax, (aname, ax) in enumerate(axs.items()):
            logger.info(f"Plot {metric}/{aname}")
            toplot, varn = aname.split("/")

            idx = np.ones(len(perf)).astype(bool)
            for vn, value in refs.items():
                if vn == varn:
                    continue
                idx &= perf.loc[:, f"TASK_OPT_{vn}"] == value

            df = perf.loc[idx]
            opts = df.filter(regex="TASK_OPT", axis=1).drop_duplicates()

            df = pd.pivot_table(df,
                                index=("TASK_OPT_stationid", "REPEAT"),
                                columns=f"TASK_OPT_{varn}",
                                values=f"{toplot}_{metric}_quasoare")
            try:
                vl = violinplot.Violin(df)
            except:
                continue
            vl.draw(ax=ax)

            # Axis
            title = f"({letters[iax]}) {aname}"
            ax.set(title=title,
                   xlabel="",
                   ylabel=metric)

        # Save file
        fp = script_paths.fimg / \
                f"figA_gradbench_{metric}_v{config.version}.{config.imgext}"
        fig.savefig(fp, dpi=config.fdpi, transparent=config.ftransparent)
        #putils.blackwhite(fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table presenting gradient benchmarking",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version",
                        help="Version number",
                        type=str, required=True)
    args = parser.parse_args()

    # Config
    version = args.version
    awidth = 6
    aheight = 5
    fdpi = 300
    imgext = "png"
    ftransparent = False

    Config = namedtuple("Config", ["version", "awidth",
                                   "aheight", "fdpi",
                                   "ftransparent", "imgext"])
    config = Config(version, awidth, aheight, fdpi,
                    ftransparent, imgext)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
