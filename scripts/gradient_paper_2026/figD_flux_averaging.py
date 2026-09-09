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
from string import ascii_lowercase as letters

import numpy as np
import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker as mticker
from matplotlib.patches import Rectangle

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils
from hydrodiy.stat import metrics

from pyquasoare import gradients
import c_pyquasoare_grad
from gradbench import data_reader

import figC_gr4j_gradient
import importlib
importlib.reload(figC_gr4j_gradient)

from figC_gr4j_gradient import get_data, configure_model
from figC_gr4j_gradient import set_parameters, get_parameters

# Select backend
#mpl.use("Agg")

def aggregate(x, ntimesteps, nagg):
    agg = x.cumsum(axis=0)
    agg[nagg:] = agg[nagg:] - agg[:-nagg]
    idx = np.arange(nagg - 1, (ntimesteps - 1) * nagg, nagg)
    return agg[idx] / nagg

def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fimg = froot / "images" / "manuscript" / "gr4j_flux_averaging"
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

    # params
    X0, X1, X2, X3, X4 = get_parameters()

    # Prepare model
    fluxes = {}
    dfluxes = {}
    stores = {}
    mods = {}
    ntimesteps = data.hydroclim.shape[0]
    #nsubs = [1, 24, 24 * 6, 24 * 60]
    nsubs = [1, 6, 12]

    for nsub in nsubs:
        config = config._replace(nsubsteps=nsub)
        mod = configure_model(logger, config, data)
        set_parameters(mod, nsub, X0, X1, X2, X3, X4)
        mod.run()
        logger.info(f"nts = {mod.ntimesteps}", ntab=1)
        logger.info(f"ts = 1/{1./mod.timestep:0.0f}", ntab=1)
        logger.info(f"fs3[3] = {mod.flux_scalings[3, 2]:0.2f}", ntab=1)
        mods[nsub] = mod

    nres = mod.nreservoirs
    mosaic = [[f"fx/Qres{res}", f"dfx/Qres{res}", f"x/Qres{res}"]
              for res in [0, 1, nres - 2, nres - 1]]
    nrows, ncols = len(mosaic), len(mosaic[0])
    figsize = (config.awidth * ncols, config.aheight * nrows)
    plt.close("all")
    fig = plt.figure(figsize=figsize, layout="constrained")
    axs = fig.subplot_mosaic(mosaic, sharex=True)

    time = data.hydroclim.index[:-1]
    ivar = 3

    for iax, (aname, ax) in enumerate(axs.items()):
        ptype, varn = aname.split("/")
        ires = int(re.sub("Qres", "", varn))

        df = {}
        for nsub, mod in mods.items():
            if ptype == "fx":
                q = aggregate(mod.fluxes[:, ires, 5], ntimesteps,
                              nsub)
                df[nsub] = pd.Series(q, index=time)
            elif ptype == "x":
                k = np.arange(nsub - 1, nsub * (ntimesteps - 1), nsub)
                x = mod.x[k, ires]
                df[nsub] = pd.Series(x, index=time)
            else:
                dq = aggregate(mod.dfluxes[:, ivar, ires, 5], ntimesteps,
                               nsub)
                df[nsub] = pd.Series(dq, index=time)

        df = pd.DataFrame(df)
        df.plot(ax=ax)
        if ptype != "x":
            ax.set_yscale("asinh")
            y0, y1 = ax.get_ylim()
            ax.set_ylim((y1, y0))
        title = f"Reservoir {ires} / {ptype}"
        ax.set_title(title)

        tax = ax.twinx()
        diff = df.iloc[:, [0, -1]].diff(axis=1).iloc[:, -1]
        diff.plot(ax=tax, color="0.5")
        ax.plot([], [], "-", color="0.5", label="diff")


    fp = script_paths.fimg / "gr4j_flux_aggregation.png"
    fig.savefig(fp)

    logger.completed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table presenting gradient benchmarking",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--stationid",
                        help="StationID",
                        type=str, default="203014")
    parser.add_argument("-d", "--debug",
                        help="Debug mode",
                        action="store_true", default=False)
    parser.add_argument("-n", "--nuh",
                        help="Number of UH reservoirs",
                        type=int, default=5)
    parser.add_argument("-z", "--zoom",
                        help="Time zoom level",
                        type=int, default=1)
    args = parser.parse_args()

    # Config
    timestep = "daily"

    debug = args.debug
    nalphas = 50
    nuh = args.nuh
    zoom = args.zoom
    nsubsteps = 1
    start = "2010"
    end = "2023"

    stationid = args.stationid
    awidth = 7
    aheight = 3
    fdpi = 300
    imgext = "png"
    ftransparent = False

    Config = namedtuple("Config", ["awidth",
                                   "aheight", "fdpi",
                                   "ftransparent", "imgext",
                                   "timestep", "stationid",
                                   "nalphas", "nuh", "zoom",
                                   "nsubsteps", "start", "end",
                                   "debug"])
    config = Config(awidth, aheight, fdpi,
                    ftransparent, imgext,
                    timestep, stationid,
                    nalphas, nuh, zoom,
                    nsubsteps, start, end,
                    debug)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
