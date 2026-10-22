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

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils
from hydrodiy.stat import metrics

from pyquasoare import gradients
import c_pyquasoare_grad
from gradbench import data_reader

import importlib
import figC_grad_diag_structure
importlib.reload(figC_grad_diag_structure)

from figC_grad_diag_structure import get_data, configure_model, set_parameters
from figB_reservoir_example import format_time_xaxis
from utils import get_parameters

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
    fimg = froot / "images" / "manuscript" / "figD_gr4j_flux_averaging"
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
    #nsubs = [1, 2, 4, 8, 12, 24, 72]
    nsubs = [1, 2, 4, 24]

    for nsub in nsubs:
        config = config._replace(nsubsteps=nsub)
        mod = configure_model(logger, config, data)
        set_parameters(mod, nsub, X0, X1, X2, X3, X4)
        mod.run()
        mod.aggregate()
        logger.info(f"nsub= {mod.nsubsteps}", ntab=1)
        logger.info(f"nts = {mod.ntimesteps}", ntab=1)
        logger.info(f"ts = 1/{1./mod.timestep:0.0f}", ntab=1)
        logger.info(f"fs3[3] = {mod.flux_scalings[3, 2]:0.2f}", ntab=1)
        mods[nsub] = mod

    nres = mod.nreservoirs
    mosaic = [[f"{vn}/Qres{res}" for res in [2, nres - 1]]
              for vn in ["fx", "dfx"]]

    nrows, ncols = len(mosaic), len(mosaic[0])
    figsize = (config.awidth * ncols, config.aheight * nrows)
    plt.close("all")
    fig = plt.figure(figsize=figsize, layout="constrained")
    axs = fig.subplot_mosaic(mosaic, sharex=True)

    time_agg = data.hydroclim.index
    ivar = 2

    for iax, (aname, ax) in enumerate(axs.items()):
        ptype, varn = aname.split("/")
        ires = int(re.sub("Qres", "", varn))

        leg_agg = []
        leg_raw = []
        for nsub, mod in mods.items():
            dt = pd.Timedelta(minutes=(24 * 60)//nsub)
            nts = mod.fluxes.shape[0]
            time = pd.date_range(time_agg[0], freq=dt, periods=nts)

            if ptype == "fx":
                se = pd.Series(mod.fluxes[:, ires, 5], index=time)
                se_agg = pd.Series(mod.fluxes_agg[:, ires, 5], index=time_agg)
            elif ptype == "x":
                se = pd.Series(mod.x[:, ires], index=time)
                k = mod.idx_end_timestep_agg
                se_agg = pd.Series(mod.x[k, ires], index=time_agg)
            else:
                se = pd.Series(mod.dfluxes[:, ivar, ires, 5], index=time)
                se_agg = pd.Series(mod.dfluxes_agg[:, ivar, ires, 5],
                                   index=time_agg)

            col = "k" if nsub == 1 else None
            lw = 3 if nsub == 1 else 2
            dt = 24 / nsub
            lab = f"{dt:2.0f}h daily average"
            ax.step(se_agg.index, se_agg.values, where="post",
                    color=col, label=lab, lw=lw)
            line = ax.get_lines()[-1]
            col = line.get_color()
            leg_agg.append(line)
            if nsub > 1:
                lab = f"{dt:2.0f}h simulation"
                ax.plot(se.index, se.values, "--", color=col, label=lab)
                line = ax.get_lines()[-1]
                leg_raw.append(line)

        if ptype != "x":
            #ax.set_yscale("asinh")
            y0, y1 = ax.get_ylim()
            ax.set_ylim((y1, y0))

        if ires == 2:
            rname = "First UH store"
        elif ires == nres - 1:
            rname = "Routing store"
        else:
            rname = "?"

        if ptype == "fx":
            yname = "Outflow flux"
            yunit = "[mm.day-1]"
        elif ptype == "dfx":
            yname = "Outflow flux gradient"
            yunit = "[day-1]"

        format_time_xaxis(ax)
        title = f"({letters[iax]}) {yname} - {rname} (reservoir #{ires + 1})"
        ylab = f"{yname} {yunit}"
        ax.set(title=title, ylabel=ylab)
        if iax == 0:
            ax.legend(handles=leg_agg + leg_raw,
                      loc=1, framealpha=1.)

        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    fp = script_paths.fimg / "gr4j_flux_aggregation.png"
    fig.savefig(fp, dpi=config.fdpi)

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
    start = "2022-02-26"
    end = "2022-03-04"

    stationid = args.stationid
    awidth = 5.5
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
