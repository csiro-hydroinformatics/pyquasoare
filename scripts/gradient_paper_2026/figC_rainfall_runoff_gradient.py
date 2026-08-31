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
    fimg = froot / "images" / "manuscript" / "gr4j_gradient"
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


def get_data(config, script_paths, logger):
    stationid = config.stationid
    logger.info(f"Load data for stations {stationid}")
    stations = data_reader.get_stations()
    area = stations.loc[stationid, "CATCHMENTAREA[km2]"]

    timestep = config.timestep
    hydroclim = data_reader.get_data(stationid, timestep, nval=None)
    hydroclim = hydroclim.loc["2015":]

    hydroclim = hydroclim.rename(columns={
        "RAINFALL[mm_day-1]": "P",
        "PET[mm_day-1]": "PET",
        "STREAMFLOW[mm_day-1]": "Q"
        })
    hydroclim = hydroclim.loc[:, ["P", "PET", "Q"]]

    data = namedtuple("Data", ["hydroclim", "area"])(hydroclim, area)
    logger.info(".. done")

    return data


def configure_model(logger, config, data):
    logger.info(f"Create and configure model")
    nalphas = config.nalphas
    nuh = config.nuh
    nres = 3 + nuh
    mod = gradients.MultiReservoirModel(nres, nalphas,
                                        errors="raise")
    # Links
    # .. Interception -> prod
    mod.add_link_bypass_shortfall(0, 1)

    # .. prod -> UH
    for iuh in range(nuh):
        mod.add_link_outflow(iuh + 1, iuh + 2)

    # .. UH -> Route
    mod.add_link_outflow(nres - 2, nres - 1)

    # Coefs
    mod.set_coefs_linear(0, "openwater")
    mod.set_coefs_linear(1, "gr4jprod")
    for iuh in range(nuh):
        mod.set_coefs_linear(2 + iuh, "linrouting")
    mod.set_coefs_linear(nres - 1, "gr4jroute")

    # Allocate
    # .. indep var = prod / route capacity + exchange
    #                + scaling of inputs
    hclim = data.hydroclim.loc[:, ["P", "PET"]]
    ntimesteps = len(hclim)
    vspecs = [[0, 1], [0, 2]]\
        + [[1, 0], [nres - 1, 0], [nres - 1, 1]]
    mod.allocate(ntimesteps, variables_specs=vspecs)

    # Set inputs
    inputs = np.ascontiguousarray(hclim.values)
    inputs[:, 1] *= -1
    mod.set_Qstars(0, inputs)

    # Set demand
    inputs = np.ones((ntimesteps, 2)) * np.nan
    inputs[:, 1] = 1.
    mod.set_Qstars(nres - 1, inputs)

    return mod


def configure_gr4j(logger, config, data):
    gr = GR4J()
    hclim = data.hydroclim.loc[:, ["P", "PET"]]
    gr.allocate(hclim, noutputs=gr.noutputsmax)
    return gr


def set_parameters(mod, X0, X1, X2, X3, X4, eta_P=1., eta_E=1.):
    nres = mod.nreservoirs
    nuh = nres - 3
    mod.gammas = [X0, X1] + [1] * nuh + [X3]
    for iuh in range(nuh):
        # See Santos (2018), Eq. 6
        mod.coefs_linear[2 + iuh, 2] *= nuh * 1.25 / X4

    mod.flux_scalings[nres - 1, 1] = X2
    mod.flux_scalings[0, 0] = eta_P
    mod.flux_scalings[0, 1] = eta_E


def process(config, script_paths, logger, data):
    logger.info("Run model")
    # Prepare model
    mod = configure_model(logger, config, data)
    modb = configure_model(logger, config, data)
    gr = configure_gr4j(logger, config, data)

    # Good parameters
    X0 = 1.
    X1 = 400
    X2 = -0.5
    X3 = 100.
    X4 = 1.
    set_parameters(mod, X0, X1, X2, X3, X4)

    gr.params.values = [X1, X2, X3, X4]

    # Run model
    mod.run()
    gr.run()

    # Timeseries data
    qobs = -data.hydroclim.Q
    time = qobs.index
    qsim = pd.Series(mod.fluxes[:, -1, 5], index=time)

    grsim = gr.to_dataframe(index=time)

    # Metrics
    iok = qobs.notnull()
    qo, qs = qobs[iok], qsim[iok]
    eps = np.abs(qo.mean()) * 1e-2
    trans = {
        "id": lambda x: x,
        "sqrt": lambda x: (np.sqrt(eps + np.abs(x)) - math.sqrt(eps)) * np.sign(x),
        "log": lambda x: (np.log(eps + np.abs(x)) - math.log(eps)) * np.sign(x),
        }
    perf = {}
    for tn, t in trans.items():
        tqo, tqs = t(qo), t(qs)
        perf[f"nse_{tn}"] = metrics.nse(tqo, tqs)
        perf[f"bias_{tn}"] = metrics.bias(tqo, tqs)
    perf = pd.Series(perf).sort_index()
    logger.info(f"\n{perf}")

    # Plot
    start, end = "2015", "2022"
    #start, end = "2022-02-01", "2022-04-15"
    #start, end = "2022-02-22", "2022-03-10"

    logger.info("Plot vs GR4J")
    mosaic = [["S", "PR", "AE"], ["R", "UH", "Q"]]
    ncols, nrows = len(mosaic[0]), len(mosaic)
    figsize = (config.awidth * ncols, config.aheight * nrows)
    plt.close("all")
    fig = plt.figure(figsize=figsize, layout="constrained")
    axs = fig.subplot_mosaic(mosaic, sharex=True)
    for aname, ax in axs.items():
        if aname == "UH":
            v1 = grsim.loc[start:end, "Q1"] + grsim.loc[start:end, "Q9"]
        else:
            v1 = grsim.loc[start:end, aname]

        if aname == "S":
            v2 = mod.x[:, 1] * X1
        elif aname == "PR":
            v2 = -mod.fluxes[:, 1, 5]
        elif aname == "AE":
            v2 = -mod.fluxes[:, 0, 1] - mod.fluxes[:, 1, 1]
        elif aname == "R":
            v2 = mod.x[:, -1] * X3
        elif aname == "QR":
            v2 = -mod.fluxes[:, -1, 2]
        elif aname == "Q":
            v2 = -mod.fluxes[:, -1, 5]
        elif aname == "UH":
            v2 = -mod.fluxes[:, -2, 5]

        v2 = pd.Series(v2, index=grsim.index).loc[start:end]

        v1.plot(ax=ax, label="GR4J")
        v2.plot(ax=ax, label="Quas")
        ax.set(title=aname)

    fp = script_paths.fimg / "gr4j_comparison.png"
    fig.savefig(fp)


    logger.info("Plot grad")
    plt.close("all")

    nvars = len(mod.variables_type_id)
    nrows, ncols = nvars + 1, 1
    figsize = (config.awidth * ncols, config.aheight * nrows)
    fig, axs = plt.subplots(nrows=nrows,
                            figsize=figsize,
                            layout="constrained",
                            sharex=True)
    ax = axs[0]
    qobs[start:end].plot(ax=ax, label="Observed", color="k", ls="--")
    qsim[start:end].plot(ax=ax, label="$Q_3$", lw=2)

    sqeps = math.sqrt(eps)
    fwd = lambda x: (np.sqrt(eps + np.abs(x)) - sqeps) * np.sign(x)
    inv = lambda y: np.sign(y) * ((np.abs(y) + sqeps)**2 - eps)
    u = np.linspace(-1, 1, 100)
    assert np.allclose(inv(fwd(u)), u)

    ax.set_yscale("function", functions=(fwd, inv))
    title = "(a) Outflow of the routing reservoir $Q_3$"
    ax.set_title(title, va="top", ha="left",
                 x=0.02, y=0.94)
    ax.set(ylabel="Streamflow [mm.d$^{-1}$]")
    y0, y1 = ax.get_ylim()
    ax.set_ylim((0., y0))
    ax.legend()

    dqmax = 0.
    vnames = ["\\eta_1^f", "\\eta_2^f",
              "X_1", "X_3", "X_2"]
    for i in range(nvars):
        dq = pd.Series(mod.dfluxes[:, i, -1, 5],
                       index=qobs.index)
        dqb = pd.Series(modb.dfluxes[:, i, -1, 5],
                        index=qobs.index)

        vid = mod.variables_type_id[i]
        rid = mod.variables_reservoir_id[i]
        if vid == 0:
            dq *= mod.gammas[rid]
            dqb *= mod.gammas[rid]
            title = f"Gradients ${vnames[i]}dQ_3/d{vnames[i]}$"
        else:
            title = f"Gradients $dQ_3/d{{{vnames[i]}}}$"

        ax = axs[i + 1]
        dq[start:end].plot(ax=ax, lw=2)
        #dqb[start:end].plot(ax=ax, lw=2)

        #ax.set_yscale("function", functions=(fwd, inv))
        ax.set_yscale("asinh")

        putils.line(ax, 1, 0, 0, 0, "k-", lw=0.8)
        ax.set_title(title, va="top", ha="left",
                     x=0.02, y=0.94)
        y0, y1 = ax.get_ylim()
        ax.set(ylabel="Gradient [mm.d$^{-1}$]",
               ylim=(y1, y0))

        dqm = np.abs(dq[start:end]).max()
        if dqm > dqmax:
            imax = i
            dqmax = dqm

    for i in range(nvars):
        axs[i + 1].set_ylim((-dqmax, dqmax))

    fp = script_paths.fimg / "gr4j_fluxes.png"
    fig.savefig(fp)

    logger.completed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table presenting gradient benchmarking",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--stationid",
                        help="StationID",
                        type=str, default="203014")
    args = parser.parse_args()

    # Config
    timestep = "daily"

    nalphas = 50
    nuh = 2

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
                                   "nalphas", "nuh"])
    config = Config(awidth, aheight, fdpi,
                    ftransparent, imgext,
                    timestep, stationid,
                    nalphas, nuh)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
