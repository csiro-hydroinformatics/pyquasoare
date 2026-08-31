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
from scipy.special import gamma
from string import ascii_lowercase as letters

import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker as mticker
from matplotlib.patches import Rectangle

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils
from hydrodiy.stat import metrics

from pygme.models.gr4j import GR4J

from pyquasoare import gradients
import c_pyquasoare_grad
from gradbench import data_reader

# Select backend
#mpl.use("Agg")


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fimg = froot / "images" / "explore" / "linrouting_uh"
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


def process(config, script_paths, logger, data):
    logger.info("Run model")

    # Nash cascade
    nres = 12
    tau = 20

    nalphas = 10
    mod = gradients.MultiReservoirModel(nres, nalphas,
                                        errors="raise")
    for ires in range(nres):
        mod.set_coefs_linear(ires, "linrouting")
        if ires > 0:
            mod.add_link_outflow(ires - 1, ires)
    mod.gammas = 1.

    # Set data
    nval = int(tau * 5)
    mod.allocate(nval)
    inputs = np.zeros((nval, 2))
    inputs[0, 0] = 1.
    mod.set_Qstars(0, inputs)

    mod.x0 = 1e-6

    # Set parameters
    nu = nres
    for ires in range(nres):
        mod.coefs_linear[ires, 2] *= nu / tau

    # Run
    mod.run()

    plt.close("all")
    fig, axs = plt.subplots(ncols=4, nrows=3,
                            layout="constrained")

    for iax, ax in enumerate(axs.flat):
        if iax >= nres:
            ax.axis("off")
            continue
        fx = -mod.fluxes[:, iax, 5]
        nval = len(fx)
        t = np.arange(nval)
        ii = np.arange(0, 30)
        ax.step(t[ii], inputs[ii, 0], "k--", where="post")
        ax.step(t[ii], fx[ii], where="post")

        tt = np.linspace(t[ii][0], t[ii][-1], 1000)
        n = iax + 1
        k = n / tau
        yy = k / gamma(n) * (k * tt)**(n - 1) * np.exp(-k * tt)
        ax.plot(tt, yy)

        title = f"{n} reservoirs"
        ax.set(title=title)

    plt.show()



    logger.completed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exploration of Nash Cascade",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--stationid",
                        help="StationID",
                        type=str, default="203014")
    args = parser.parse_args()

    # Config
    awidth = 7
    aheight = 3
    fdpi = 300
    imgext = "png"
    ftransparent = False

    Config = namedtuple("Config", ["awidth",
                                   "aheight", "fdpi",
                                   "ftransparent", "imgext"])
    config = Config(awidth, aheight, fdpi,
                    ftransparent, imgext)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Process
    process(config, script_paths, logger, None)

    logger.completed()
