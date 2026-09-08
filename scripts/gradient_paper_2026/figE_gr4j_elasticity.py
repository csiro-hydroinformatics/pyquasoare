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
from scipy.optimize import minimize

import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker as mticker

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils
from hydrodiy.stat import metrics

from pygme.models.gr4j import GR4J

from pyquasoare import gradients
import c_pyquasoare_grad

import figC_gr4j_gradient
import importlib
importlib.reload(figC_gr4j_gradient)

from figC_gr4j_gradient import get_data, \
    configure_model, set_parameters

# Select backend
#mpl.use("Agg")

N_TM = 2.

def tm_runoff(p, e, n=N_TM):
    arid = p / e
    return 1 - 1./(1 + arid**n)**(1./n)

def tm_q(p, e, n=N_TM):
    return p * tm_runoff(p, e, n)

def tm_eP(p, e, n=N_TM):
    arid = p / e
    r = tm_runoff(p, e, n)
    return (1 - 1. / (1 + arid**n)**(1. + 1. / n)) / r

def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fimg = froot / "images" / "manuscript" / "gr4j_elasticity"
    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fimg"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fimg)
    fimg.mkdir(exist_ok=True, parents=True)
    for f in fimg.glob("*.png"):
        f.unlink()

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    logger = iutils.get_logger(basename, console=True)
    logger.log_dict(config._asdict(), "Configuration")
    logger.info("", nret=1)
    return logger


def format_y(ax, y0, y1):
    ax.set(ylabel="Elasticity $\\epsilon_P$ [-]",
           ylim=(y0, y1))
    #ax.grid(axis="y")
    #ax.yaxis.set_major_locator(mticker.MaxNLocator(5))


def format_x(ax, start, end):
    ybase = 5
    ax.xaxis.set_major_locator(mdates.YearLocator(base=ybase))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    s = pd.to_datetime(start + "-01-01")
    e = pd.to_datetime(end + "-12-31")
    ax.set_xlim((s, e))


def process(config, script_paths, logger, data):
    logger.info("Run model")
    # Prepare model
    mod = configure_model(logger, config, data)


    # Good parameters
    X0 = 1.
    X1 = 500.
    X2 = -0.1
    X3 = 50.
    X4 = 1.0
    set_parameters(mod, config.nsubsteps, X0, X1, X2, X3, X4)

    # Run model
    mod.run()

    # Timeseries data
    time = data.hydroclim.index
    qsim = pd.Series(mod.fluxes[:, -1, 5], index=time)

    p_interan = data.hydroclim.P.mean() * 365.25
    e_interan = data.hydroclim.PET.mean() * 365.25

    p_an = data.hydroclim.P.resample("YS").sum()
    e_an = data.hydroclim.PET.resample("YS").sum()

    # Gradients
    nuh = config.nuh
    grads = pd.Series(mod.dfluxes[:, 0, -1, 5], index=time)

    # .. inversing sign because qsim is neg
    q_an = -qsim.resample("YS").sum()
    runoff_an = q_an / p_an
    q_interan = -qsim.mean()

    elast = -grads / q_interan
    elast_mthly = elast.resample("MS").mean()
    elast_an = elast.resample("YS").mean()
    elast_interan = elast.mean()

    ## .. check
    #eps = 1e-6
    #modp = configure_model(logger, config, data)
    #modp.Qstars[:, 0, 0] *= 1 + eps
    #set_parameters(modp, config.nsubsteps, X0, X1, X2, X3, X4)
    #modp.run()

    #modm = configure_model(logger, config, data)
    #modm.Qstars[:, 0, 0] *= 1 - eps

    #set_parameters(modm, config.nsubsteps, X0, X1, X2, X3, X4)
    #modm.run()

    #dfx = (modp.fluxes[:, -1, 5] - modm.fluxes[:, -1, 5]) / 2 / eps
    #qm = -mod.fluxes[:, -1, 5].mean()
    #elastn = pd.Series(-dfx / qm, index=time)
    #elastn_an = elastn.resample("YS").mean()

    # Turc-Mezentsev -> elasticity
    runoff_an_tm = tm_runoff(p_an, e_an)
    q_an_tm = tm_q(p_an, e_an)
    elast_an_tm = tm_eP(p_an, e_an)

    runoff_interan_tm = tm_runoff(p_interan, e_interan)
    q_interan_tm = tm_q(p_interan, e_interan)
    elast_interan_tm = tm_eP(p_interan, e_interan)

    # Plot
    logger.info("Plot elast")
    plt.close("all")

    nrows, ncols = 2, 2
    mosaic = [["monthly", "scatter"], ["annual", "interan"]]
    ncols, nrows = len(mosaic[0]), len(mosaic)
    figsize = (config.awidth * ncols, config.aheight * nrows)
    fig = plt.figure(figsize=figsize,
                     layout="constrained")
    kw = dict(width_ratios=[2, 1])
    axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

    # Scatter data
    ax = axs["scatter"]
    arid = p_an / e_an
    ax.plot(arid, q_an / p_an, "+", label="Q G")
    ax.plot(arid, q_an_tm / p_an, "+", label="Q T")
    ax.legend()

    tax = ax.twinx()
    tax.plot(arid, elast_an, "o", mfc="none")
    tax.plot(arid, elast_an_tm, "o", mfc="none")

    # ... monthly data
    ax = axs["monthly"]

    t = elast.index
    v = elast
    ax.plot(t, v, color="0.5", lw=0.5, alpha=0.5)

    t = elast_mthly.index
    v = elast_mthly
    ax.plot(t, v, label="GR4J QuaSoARe")

    title = "(a) Monthly streamflow elasticity to rainfall"
    ax.set_title(title, va="top", ha="left",
                 x=0.02, y=0.94)

    format_x(ax, config.start, config.end)
    ax.set_yscale("log")
    format_y(ax, 1e-2, 1e2)

    # ... annual data
    ax = axs["annual"]

    t = elast_an.index
    v = elast_an
    ax.step(t, v, where="post", label="GR4J QuaSoARe")

    t = elast_an_tm.index
    v = elast_an_tm
    ax.step(t, v, where="post", label="Turc-Mezentsev")

    title = "(b) Annual streamflow elasticity to rainfall"
    ax.set_title(title, va="top", ha="left",
                 x=0.02, y=0.94)
    format_x(ax, config.start, config.end)
    format_y(ax, -0.2, 5)
    ax.legend(loc=3, ncol=2)

    # ... inter-annual data
    ax = axs["interan"]
    se = pd.Series({
        "GR4J\nQuaSoARe": elast_interan,
        "Turc\nMezentsev": elast_interan_tm
        })

    col1 = "tab:blue"
    col2 = "tab:orange"
    se.plot(ax=ax, kind="bar", rot=0.,
            color=[col1, col2])

    title = "(d) Inter-annual elasticity"
    ax.set_title(title, va="top", ha="left",
                 x=0.02, y=0.94)
    format_y(ax, -0.2, 5)
    ax.set(ylabel="")

    fp = script_paths.fimg / "gr4j_elasticity.png"
    fig.savefig(fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table presenting gradient benchmarking",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--stationid",
                        help="StationID",
                        type=str, default="203014")
    parser.add_argument("-n", "--nuh",
                        help="Number of UH reservoirs",
                        type=int, default=2)
    parser.add_argument("-z", "--zoom",
                        help="Time zoom level",
                        type=int, default=1)
    args = parser.parse_args()

    # Config
    timestep = "daily"

    nalphas = 50
    nuh = args.nuh
    zoom = args.zoom
    nsubsteps = 1
    start = "1998"
    end = "2001"

    stationid = args.stationid
    awidth = 5
    aheight = 3
    fdpi = 300
    imgext = "png"
    ftransparent = False

    Config = namedtuple("Config", ["awidth",
                                   "aheight", "fdpi",
                                   "ftransparent", "imgext",
                                   "timestep", "stationid",
                                   "nalphas", "nuh", "zoom",
                                   "nsubsteps", "start", "end"])
    config = Config(awidth, aheight, fdpi,
                    ftransparent, imgext,
                    timestep, stationid,
                    nalphas, nuh, zoom,
                    nsubsteps, start, end)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
