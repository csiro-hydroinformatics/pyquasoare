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
from hydrodiy.plot import putils

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

    numdiff_eps = 1e-5
    perf = perf.loc[perf.TASK_OPT_numdiff_eps == numdiff_eps]

    # Create figure
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

    for aname, ax in axs.items():
        toplot, varn = aname.split("/")

        idx = np.ones(len(perf)).astype(bool)
        for vn, value in refs.items():
            if vn == varn:
                continue
            idx &= perf.loc[:, f"TASK_OPT_{vn}"] == value

        df = perf.loc[idx].filter(regex=f"^{toplot}_", axis=1)
        import pdb; pdb.set_trace()


        # Decoration
        ax.legend(shadow=True, framealpha=0.7)

        # Axis
        title = f"{name}: Row {irow} / Column {icol}"
        ax.set_title(title)
        ax.set_xlabel("X label")
        ax.set_xlabel("Y label")

    fig.suptitle("Overall title")

    # Save file
    fp = fimg / f"image.{config.imgext}"
    fig.savefig(fp, dpi=config.fdpi, transparent=config.ftransparent)
    putils.blackwhite(fp)


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
    imext = "png"
    ftransparent = False

    Config = namedtuple("Config", ["version", "awidth",
                                   "aheight", "fdpi",
                                   "ftransparent", "imext"])
    config = Config(version, awidth, aheight, fdpi,
                    ftransparent, imext)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
