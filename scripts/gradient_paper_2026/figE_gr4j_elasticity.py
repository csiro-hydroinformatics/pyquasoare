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
from matplotlib.patches import Rectangle

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

    # Gradients
    nuh = config.nuh
    grads = pd.DataFrame(mod.dfluxes[:, :2, -1, 5], index=time)
    grads.columns = [r"$\eta_1^f$", r"$\eta_2^f$"]

    # .. inversing sign because qsim is neg
    qm = -qsim.mean()
    elast = -grads / qm

    elast_mthly = elast.resample("MS").mean()
    elast_ann = elast.resample("YS").mean()
    elast_interann = elast.mean()

    # Adim plot data
    adim = {}
    pm = data.hydroclim.P.mean()
    em = data.hydroclim.PET.mean()
    n_tm = 2.

    etaPs = np.unique([1.] + np.linspace(0.5, 2., 20).tolist())
    for etaP in etaPs:
        set_parameters(mod, config.nsubsteps, X0, X1, X2, X3, X4,
                       eta_P=etaP)
        mod.run()
        # data for the outflow from the last store
        qm = -mod.fluxes[:, -1, 5].mean()
        elast_P = -mod.dfluxes[:, 0, -1, 5].mean() / qm
        elast_E = -mod.dfluxes[:, 1, -1, 5].mean() / qm

        pmp = pm * etaP
        ratio = pmp / em
        runoff_tm = 1 - 1./(1 + ratio**n_tm)**(1. / n_tm)
        qtm = runoff_tm * pmp
        elast_P_tm = 1 - 1. / (1 + ratio**n_tm)**(1 + 1./ n_tm)
        elast_P_tm *= pmp / qtm

        elast_E_tm = - 1. / (1 + ratio**(-n_tm))**(1 + 1./ n_tm)
        elast_E_tm *= em / qtm

        adim[etaP] = {
                "aridity": pm / em * etaP,
                "runoff": qm / pm / etaP,
                "elast_P": elast_P,
                "elast_E": elast_E,
                "runoff_tm": runoff_tm,
                "elast_P_tm": elast_P_tm,
                "elast_E_tm": elast_E_tm,
        }
    adim = pd.DataFrame(adim).T

    # Turc-Mezentsev -> elasticity
    p = data.hydroclim.P.resample("YS").sum()
    e = data.hydroclim.PET.resample("YS").sum()
    qtm = 1 - 1./(1 + (p / e)**n_tm)**(1. / n_tm)
    elast_ann_tm = 1 - 1. / (1 + (p / e)**n_tm)**(1. + 1. / n_tm)
    elast_ann_tm *= 1 / qtm

    # Plot
    logger.info("Plot elast")
    plt.close("all")

    nrows, ncols = 2, 2
    mosaic = [["monthly", "."], ["annual", "interan"]]
    ncols, nrows = len(mosaic[0]), len(mosaic)
    figsize = (config.awidth * ncols, config.aheight * nrows)
    fig = plt.figure(figsize=figsize,
                     layout="constrained")
    kw = dict(width_ratios=[2, 1])
    axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

    # ... monthly data
    ax = axs["monthly"]

    elast_mthly.iloc[:, 0].plot(ax=ax, label="GR4J QuaSoARe")

    title = "(a) Monthly streamflow elasticity to rainfall"
    ax.set_title(title, va="top", ha="left",
                 x=0.02, y=0.94)
    _, y1 = ax.get_ylim()
    fwd = lambda x: np.sqrt(np.abs(x)) * np.sign(x)
    inv = lambda y: np.sign(y) * y**2

    def format_y(ax):
        ax.set(ylabel="Elasticity $\epsilon_P$ [-]",
               ylim=(-0.1, y1))
        #ax.set_yscale("function", functions=(fwd, inv))
        ax.set_yscale("log")
        ax.set_yticks([0.1, 1., 2., 5., 10.])
        ax.grid(axis="y")

    format_y(ax)

    # ... annual data
    ax = axs["annual"]

    elast_ann.iloc[:, 0].plot(ax=ax, label="GR4J QuaSoARe")
    elast_ann_tm.plot(ax=ax, label="Turc-Mezentsev")

    title = "(b) Annual streamflow elasticity to rainfall"
    ax.set_title(title, va="top", ha="left",
                 x=0.02, y=0.94)
    format_y(ax)
    ax.legend(loc=3, ncol=2)

    axi = ax.inset_axes((0.78, 0.6, 0.2, 0.38))
    axi.plot(elast_ann_tm, elast_ann.iloc[:, 0], "o")

    # .. adimensional plot
    #ax = axs["adim"]
    #putils.waterbalplot(ax, ncoeff=n_tm)

    #ax.plot(adim.aridity, adim.runoff, "-", label="GR4J-QuaSoARe")
    #col1 = ax.get_lines()[-1].get_color()
    #ax.plot(adim.aridity[1.], adim.runoff[1.], "o", color=col1)

    #ax.plot(adim.aridity, adim.runoff_tm, "-", label="TM")
    #col2 = ax.get_lines()[-1].get_color()
    #ax.plot(adim.aridity[1.], adim.runoff_tm[1.], "o", color=col2)

    #tax = ax.twinx()
    #tax.plot(adim.aridity, adim.elast_P, "--", color=col1)
    #tax.plot(adim.aridity[1.], adim.elast_P[1.], "o", color=col1)

    #tax.plot(adim.aridity, adim.elast_P_tm, "--", color=col2)
    #tax.plot(adim.aridity[1.], adim.elast_P_tm[1.], "o", color=col2)

    #title = "(c) Adimensional elasticity plot"
    #ax.set_title(title, va="top", ha="left",
    #             x=0.02, y=0.94)

    # ... inter-annual data
    ax = axs["interan"]
    se = pd.Series({
        "GR4J\nQuaSoARe": adim.loc[1., "elast_P"],
        "Turc\nMezentsev": adim.loc[1., "elast_P_tm"]
        })

    col1 = "tab:blue"
    col2 = "tab:orange"
    se.plot(ax=ax, kind="bar", rot=0.,
            color=[col1, col2])

    title = "(d) Inter-annual elasticity"
    ax.set_title(title, va="top", ha="left",
                 x=0.02, y=0.94)
    format_y(ax)
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
    start = "1980"
    end = "2022"

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
