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

    # Get data
    perf = data.perf

    idx = perf.TASK_OPT_gamma_mean == "500_50_500"
    idx &= perf.TASK_OPT_nalphas == 100
    idx &= perf.TASK_OPT_stationid == 203014
    perf = perf.loc[idx]

    epss = perf.TASK_OPT_numdiff_log10eps.unique()
    for eps in epss:
        logger.info(f"Plotting eps = {eps}", nret=1)

        # Create figure
        plt.close("all")
        nrs = [1, 5, 50] if eps == -5 else [5]
        metrics = ["runtime", "errmax00", "errmax90"]
        if eps != -5:
            metrics = metrics[1:]
        mosaic = [[f"{nr}/{vn}" for nr in nrs]
                  for vn in metrics]
        fncols, fnrows = len(mosaic[0]), len(mosaic)
        awidth, aheight = config.awidth, config.aheight
        figsize = (awidth*fncols, aheight*fnrows)
        fig = plt.figure(constrained_layout=True,
                         figsize=figsize)

        #gw = dict(height_ratios=[1]*fnrows, width_ratios=[1]*fncols)
        #axs = fig.subplot_mosaic(mosaic, gridspec_kw=gw)
        axs = fig.subplot_mosaic(mosaic)

        nstations = len(perf.TASK_OPT_stationid.unique())
        nrepeat = len(perf.REPEAT.unique())

        for iax, (aname, ax) in enumerate(axs.items()):
            nreaches, metric = aname.split("/")
            nreaches = int(nreaches)
            logger.info(f"Plot reaches={nreaches}/{metric}", ntab=1)

            idx = perf.TASK_OPT_numdiff_log10eps == eps
            idx &= perf.TASK_OPT_nreaches == nreaches
            pat = f"{metric}"
            cc = [cn for cn in perf.columns if re.search(pat, cn)]
            df = perf.loc[idx, cc]
            logger.info(f"len(df) = {len(df)}", ntab=1)

            if len(df) != nrepeat * nstations:
                logger.info("no data. continue", ntab=1)
                ax.axis("off")
                continue

            try:
                vl = violinplot.Violin(df)
            except:
                continue
            vl.draw(ax=ax)

            # Axis
            title = f"({letters[iax]}) {nreaches} reaches - {metric}"
            ax.set(title=title,
                   xlabel="",
                   ylabel=metric)

            if re.search("log10", cc[0]):
                ax.set_ylim((-5.5, 1.5))
                tk = np.arange(-5, 1)
                labs = ["1" if k == 0 else f"$10^{{{k}}}$" for k in tk]
                ax.set_yticks(tk, labels=labs)

            ax.grid(axis="y")

        # Save file
        fp = script_paths.fimg / \
                f"figA_gradbench_v{config.version}_log10eps{eps}.{config.imgext}"
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
