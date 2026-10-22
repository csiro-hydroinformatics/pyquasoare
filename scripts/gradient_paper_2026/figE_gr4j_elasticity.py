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

from pyquasoare import gradients

from figC_grad_diag_structure import get_data, configure_model, set_parameters
from utils import get_parameters

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
    fimg = froot / "images" / "manuscript" / source_file.stem
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


def format_y(ax, y0, y1, tk=None, ylabel=None):
    if ylabel is None:
        ylabel = "Elasticity $\\epsilon_{Q/P}$ [-]"

    ax.set(ylabel=ylabel,
           ylim=(y0, y1))

    if tk is not None:
        ax.set_yticks(tk)
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
    X0, X1, X2, X3, X4 = get_parameters()
    set_parameters(mod, config.nsubsteps, X0, X1, X2, X3, X4)

    # Run model
    mod.run()
    mod.aggregate()

    # Timeseries data
    time = data.hydroclim.index
    q = pd.Series(-mod.fluxes_agg[:, -1, 5], index=time)

    p_interan = data.hydroclim.P.mean() * 365.25
    e_interan = data.hydroclim.PET.mean() * 365.25
    arid_interan = p_interan / e_interan

    p_an = data.hydroclim.P.resample("YS").sum()
    e_an = data.hydroclim.PET.resample("YS").sum()
    arid_an = p_an / e_an

    p_mthly = data.hydroclim.P.resample("MS").sum()
    e_mthly = data.hydroclim.PET.resample("MS").sum()
    arid_mthly = p_mthly / e_mthly

    # Gradients
    nuh = config.nuh
    # .. inversing sign because qsim is neg
    grads = pd.Series(-mod.dfluxes_agg[:, 0, -1, 5], index=time)

    # Compute data at longer timesteps
    q_mthly = q.resample("MS").mean()
    q_an = q.resample("YS").mean()

    runoff_an = q_an / p_an
    q_interan = q.mean()

    elast_mthly = grads.resample("MS").mean() / q_mthly
    elast_an = grads.resample("YS").mean() / q_an
    elast_interan = grads.mean() / q_interan

    # Turc-Mezentsev -> elasticity
    runoff_an_tm = tm_runoff(p_an, e_an)
    q_an_tm = tm_q(p_an, e_an)
    elast_an_tm = tm_eP(p_an, e_an)

    runoff_interan_tm = tm_runoff(p_interan, e_interan)
    q_interan_tm = tm_q(p_interan, e_interan)
    elast_interan_tm = tm_eP(p_interan, e_interan)

    # Plot
    col_mthly = "lightsteelblue"
    col_an = "cornflowerblue"
    col_interan = "royalblue"

    col_tm = "tab:orange"
    ylim = (-0.3, 5.3)
    tk = np.arange(0, 6, 2)

    logger.info("Plot elast")
    plt.close("all")

    nrows, ncols = 2, 2
    mosaic = [["interan", "annual"], ["adim", "monthly"]]
    ncols, nrows = len(mosaic[0]), len(mosaic)
    figsize = (config.awidth * ncols, config.aheight * nrows)
    fig = plt.figure(figsize=figsize,
                     layout="constrained")
    kw = dict(width_ratios=[1, 1.6])
    axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

    # Scatter data
    ax = axs["adim"]

    ax.plot(arid_mthly, elast_mthly, "o", ms=4,
            label="GR4J Monthly", mec="0.8",
            color=col_mthly)

    ax.plot(arid_an, elast_an, "o", ms=6,
            label="GR4J Annual", mec="0.5",
            color=col_an)

    ax.plot(arid_interan, elast_interan, "o", ms=8,
            label="GR4J Interan.", markeredgewidth=2,
            mec="k", mfc=col_interan)

    x = np.linspace(0.1, 10., 200)
    y = tm_eP(x, np.ones_like(x))
    ax.plot(x, y, "-", label="TM",
            color=col_tm, lw=3)

    ax.legend(loc=1)
    ax.set(xlabel="Humidity $P/E_0$ [-]",
           xlim=(-0.3, 6.3))

    format_y(ax, ylim[0], ylim[1], tk)

    title = "(c) Dimensionless\n      plot"
    ax.set_title(title, va="top", ha="left",
                 x=0.02, y=0.94)

    # ... monthly data
    ax = axs["monthly"]

    t = elast_mthly.index
    v = elast_mthly
    ax.plot(t, v, label="GR4J QuaSoARe", color=col_mthly,
            lw=2)

    title = "(d) Monthly streamflow elasticity to rainfall"
    ax.set_title(title, va="top", ha="left",
                 x=0.02, y=0.94)

    format_x(ax, config.start, config.end)
    format_y(ax, ylim[0], ylim[1], tk, "")
    ax.legend(loc=3, ncol=2, framealpha=0.)

    # ... annual data
    ax = axs["annual"]

    t = elast_an.index
    v = elast_an
    ax.step(t, v, where="post", label="GR4J QuaSoARe",
            color=col_an, lw=2)
    #putils.line(ax, 1, 0, 0, v.mean(), ls="--", color=col_an)

    t = elast_an_tm.index
    v = elast_an_tm
    ax.step(t, v, where="post", label="Turc-Mezentsev",
            color=col_tm, lw=2)

    #putils.line(ax, 1, 0, 0, v.mean(), ls="--", color=col_tm)

    title = "(b) Annual streamflow elasticity to rainfall"
    ax.set_title(title, va="top", ha="left",
                 x=0.02, y=0.94)
    ax.legend(loc=3, ncol=2, framealpha=0.)
    format_x(ax, config.start, config.end)
    format_y(ax, ylim[0], ylim[1], tk, "")

    # ... inter-annual data
    ax = axs["interan"]
    se = pd.Series({
        "GR4J\nQuaSoARe": elast_interan,
        "Turc\nMezentsev": elast_interan_tm
        })

    se.plot(ax=ax, kind="bar", rot=0.,
            color=[col_interan, col_tm])

    for ivn, (nm, v) in enumerate(se.items()):
        vv = ylim[0] + (v - ylim[0]) / 2
        ax.text(ivn, vv, f"{v:0.1f}",
                va="center", ha="center",
                fontweight="bold",
                color="w")

    title = "(a) Interannual elasticity"
    ax.set_title(title, va="top", ha="left",
                 x=0.02, y=0.94)
    format_y(ax, ylim[0], ylim[1], tk)

    fp = script_paths.fimg / f"{script_paths.source_file.stem}.png"
    fig.savefig(fp, dpi=config.fdpi)


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
    nsubsteps = 6
    start = "1990"
    end = "2023"

    stationid = args.stationid
    awidth = 4.5
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
