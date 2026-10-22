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
from matplotlib import patheffects

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

from pygme.models.gr4j import GR4J

from pyquasoare import gradients
import c_pyquasoare_grad
from gradbench import data_reader

import importlib
import utils
importlib.reload(utils)

from utils import perf, get_parameters
from figC_grad_diag_structure import get_data, get_logger
from figC_grad_diag_structure import configure_model, configure_gr4j, set_parameters

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

def set_title(ax, title):
    pa = patheffects.withStroke(linewidth=6, foreground="w")
    ax.set_title(title, va="top", ha="left",
             x=0.02, y=0.93,
             path_effects=[pa])

def process(config, script_paths, logger, data):
    logger.info("Run model")
    # Prepare model
    mod = configure_model(logger, config, data)
    gr = configure_gr4j(logger, config, data)

    # Good parameters
    X0, X1, X2, X3, X4 = get_parameters()
    set_parameters(mod, config.nsubsteps, X0, X1, X2, X3, X4)
    logger.info(f"X1={X1:0.0f} X2={X2:0.1f} X3={X3:0.0f} X4={X4:0.1f}")

    gr.params.values = [X1, X2, X3, X4]

    # Run model
    mod.run()
    mod.aggregate()

    gr.run()

    # Timeseries data
    qobs = data.hydroclim.Q

    time = qobs.index
    qsim = pd.Series(-mod.fluxes_agg[:, -1, 5], index=time)

    grsim = gr.to_dataframe(index=time)

    # Gradients
    nuh = config.nuh
    grads = pd.DataFrame(mod.dfluxes_agg[:, :-nuh, -1, 5], index=time)
    grads.columns = [r"$\eta_1^f$", r"$\eta_2^f$",
                     r"$X_1$", r"$X_2$", r"$X_3$"]
    grads.loc[:, r"$X_4$"] = -mod.dfluxes_agg[:, -nuh:, -1, 5].sum(axis=1) * nuh * 1.25 * X4**2
    grads.loc[:, r"$X_2$"] /= X2
    grads = grads.filter(regex="X_\\d", axis=1)

    # Plot
    if args.zoom == 1:
        start, end = "1990", "2022"
    elif args.zoom == 2:
        start, end = "2021", "2022"
    elif args.zoom == 3:
        start, end = "2022-02-01", "2022-04-15"
    elif args.zoom == 4:
        start, end = "2022-02-22", "2022-03-10"

    # Metrics
    iok = qobs.notnull()
    qo, qs = qobs[iok][start:end], qsim[iok][start:end]
    perf_df = perf(qo, qs)
    perf_df.columns = ["Bias [-]", "NSE [-]"]
    perf_df.index = ["No transform", "sqrt(Q)", "log(Q)"]
    perf_df.index.name = "Data Transform"
    fmt = lambda x: f"{x:0.2f}"
    df = {}
    for cn, se in perf_df.items():
        df[cn] = se.apply(lambda x: f"{x:0.2f}")
    perf_df = pd.DataFrame(df)
    logger.info(f"\n{perf_df}")

    # --------------------------------------------------
    logger.info("Plot grad ts")
    plt.close("all")

    nvars = grads.shape[1]
    nrows, ncols = nvars + 1, 2
    mosaic = [["flux", "perf"]]
    #cols = grads.columns
    #mosaic += [[f"{cols[ir * 2 + ic]}" for ic in range(ncols)]
    #            for ir in range(2)]
    mosaic += [[cn for cn in grads.columns if re.search("1|3", cn)]]

    ncols, nrows = len(mosaic[0]), len(mosaic)
    figsize = (config.awidth * ncols, config.aheight * nrows)
    fig = plt.figure(figsize=figsize,
                     layout="constrained")
    kw = None #dict(width_ratios=[3, 2])
    axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw,
                             sharex=True)
    ax = axs["perf"]
    ax.axis("off")
    nc = perf_df.shape[1]
    tb = pd.plotting.table(ax, perf_df,
                           loc="center",
                           cellLoc="center",
                           bbox=[0.3, 0.25, 0.6, 0.6])

    y1 = qsim.index[0].year
    y2 = qsim.index[-1].year
    title = f"(b) Simulation performance {y1}-{y2}"
    set_title(ax, title)

    ax = axs["flux"]
    qobs[start:end].plot(ax=ax, label="Observed", color="k", ls="--")
    lab = f"GR4J\nX1={X1:0.0f} X2={X2:0.1f} X3={X3:0.0f} X4={X4:0.1f}"
    qsim[start:end].plot(ax=ax, label=lab, lw=2)

    eps = qobs.mean() * 1e-2
    sqeps = math.sqrt(eps)
    fwd = lambda x: (np.sqrt(eps + np.abs(x)) - sqeps) * np.sign(x)
    inv = lambda y: np.sign(y) * ((np.abs(y) + sqeps)**2 - eps)
    u = np.linspace(-1, 1, 100)
    assert np.allclose(inv(fwd(u)), u)

    ax.set_yscale("function", functions=(fwd, inv))
    title = "(a) Streamflow timeseries"
    #ax.set_title(title, va="top", ha="left",
    #             x=0.02, y=0.94)
    ax.set(ylabel="Streamflow [mm.d$^{-1}$]")
    set_title(ax, title)
    ax.legend(loc=3, ncol=2, framealpha=0.,
              fontsize="small")

    # Plotting gradients
    for iplot, (vn, dq) in enumerate(grads.items()):
        if vn not in mosaic[1]:
            continue
        logger.info(f"Plot {iplot} ({vn})", ntab=1)
        ax = axs[vn]

        dq[start:end].plot(ax=ax)

        #ax.set_yscale("asinh")
        putils.line(ax, 1, 0, 0, 0, "k-", lw=0.8)

        vtxt = re.sub("\\$", "", vn)
        title = f"({letters[iplot + 2]}) Gradient" \
                + f" $dQ/d{{{vtxt}}}$"
        #t = ax.set_title(title, va="top", ha="left",
        #                 x=0.02, y=0.92)
        #t.set_bbox(dict(alpha=1.0, ec="none", fc="w"))

        if re.search("1|3", vn):
            unit = "$d^{-1}$"
        elif re.search("2", vn):
            unit = "$mm.d^{-1}$"
        elif re.search("4", vn):
            unit = "$mm.d^{-2}$"
        ax.set(ylabel=f"Gradient [{unit}]")
        set_title(ax, title)
        #ax.yaxis.set_minor_locator(mticker.NullLocator())

    fp = script_paths.fimg / f"{script_paths.source_file.stem}.png"
    fig.savefig(fp, dpi=config.fdpi)

    logger.completed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table presenting gradient benchmarking",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--stationid",
                        help="StationID",
                        type=str, default="203014")
    parser.add_argument("-n", "--nuh",
                        help="Number of UH reservoirs",
                        type=int, default=5)
    parser.add_argument("-z", "--zoom",
                        help="Time zoom level",
                        type=int, default=2)
    args = parser.parse_args()

    # Config
    timestep = "daily"

    nalphas = 50
    nuh = args.nuh
    zoom = args.zoom
    nsubsteps = 1
    start = "1980"
    end = "2022"

    stationid = args.stationid
    awidth = 4.5
    aheight = 2.5
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
