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
from gradbench import data_reader

# Select backend
#mpl.use("Agg")

def smooth_nodes(x, npts=5, prc=10):
    x0, x1 = np.nanpercentile(x, [prc, 100 - prc])
    return np.linspace(x0, x1, npts)

def smooth_X(x, nodes):
    X = np.empty((len(x), len(nodes) + 1))
    X[:, 0] = 1.
    for i, n in enumerate(nodes):
        dx = x - n
        X[:, i + 1] = np.abs(dx) * dx
    return X

def smooth_reg(x, y):
    nodes = smooth_nodes(x)
    iok = ~np.isnan(x) & ~np.isnan(y)
    X = smooth_X(x[iok], nodes)

    # Quantile reg objfun
    def ofun(theta):
        yhat = X @ theta
        return np.abs(yhat - y[iok]).sum()

    theta0 = np.zeros(X.shape[1])
    theta0[0] = np.nanmean(y) - np.nanmean(x)
    opt = minimize(ofun, theta0)
    theta = opt.x

    xx = np.linspace(x[iok].min(), x[iok].max(), 500)
    XX = smooth_X(xx, nodes)
    yy = XX @ theta
    return xx, yy

def perf(qo, qs):
    eps = np.abs(qo.mean()) * 1e-2
    trans = {
        "raw": lambda x: x,
        "sqrt": lambda x: (np.sqrt(eps + np.abs(x)) - math.sqrt(eps)) * np.sign(x),
        "log": lambda x: (np.log(eps + np.abs(x)) - math.log(eps)) * np.sign(x),
        }
    perf = pd.DataFrame(index=trans.keys(), columns=["BIAS", "NSE"])
    for tn, t in trans.items():
        tqo, tqs = t(qo), t(qs)
        perf.loc[tn, "NSE"] = metrics.nse(tqo, tqs)
        perf.loc[tn, "BIAS"] = metrics.bias(tqo, tqs)
    return perf

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
    for f in fimg.glob("*.png"):
        f.unlink()

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
    hydroclim = hydroclim.loc[config.start:config.end]
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
    nsubsteps = config.nsubsteps

    # Create model
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

    # Correct for timestep related to Qstar3
    mod.flux_scalings[:, 2] = 1. / nsubsteps

    # Allocate
    # .. indep var = prod / route capacity + exchange
    #                + scaling of inputs
    hclim = data.hydroclim.loc[:, ["P", "PET"]]
    ntimesteps = len(hclim) * nsubsteps
    vspecs = [[0, 1], [0, 2], [0, 0]] \
        + [[1, 0], [nres - 1, 1], [nres - 1, 0]] \
        + [[2 + iuh, 3] for iuh in range(nuh)]
    mod.allocate(ntimesteps, variables_specs=vspecs,
                 timestep=1. / nsubsteps)

    # Set inputs
    inputs = np.repeat(np.ascontiguousarray(hclim.values),
                       nsubsteps, axis=0)
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


def set_parameters(mod, nsubsteps, X0, X1, X2, X3, X4, eta_P=1., eta_E=1.):
    nres = mod.nreservoirs
    nuh = nres - 3
    mod.gammas = [X0, X1] + [1] * nuh + [X3]

    # See Santos (2018), Eq. 6
    fs3 = nuh * 1.25 / X4
    #  + correct for timestep related to Qstar3
    fs3 *= 1. / nsubsteps

    for iuh in range(nuh):
        mod.flux_scalings[2 + iuh, 2] = fs3

    mod.flux_scalings[nres - 1, 1] = X2
    mod.flux_scalings[0, 0] = eta_P
    mod.flux_scalings[0, 1] = eta_E


def process(config, script_paths, logger, data):
    logger.info("Run model")
    # Prepare model
    mod = configure_model(logger, config, data)
    gr = configure_gr4j(logger, config, data)

    # Good parameters
    X0 = 1.
    X1 = 500.
    X2 = -0.1
    X3 = 50.
    X4 = 1.
    set_parameters(mod, config.nsubsteps, X0, X1, X2, X3, X4)

    gr.params.values = [X1, X2, X3, X4]

    # Run model
    mod.run()
    gr.run()

    # Timeseries data
    qobs = data.hydroclim.Q

    time = qobs.index
    qsim = pd.Series(-mod.fluxes[:, -1, 5], index=time)

    grsim = gr.to_dataframe(index=time)

    # Gradients
    nuh = config.nuh
    grads = pd.DataFrame(mod.dfluxes[:, :-nuh, -1, 5], index=time)
    grads.columns = [r"$\eta_1^f$", r"$\eta_2^f$",
                     r"$X_0$", r"$X_1$", r"$X_2$", r"$X_3$"]
    #grads.loc[:, r"$X_4$"] = -mod.dfluxes[:, -nuh:, -1, 5].sum(axis=1) * nuh * 1.25 / X4

    grads.loc[:, r"$X_4$"] = -mod.dfluxes[:, -nuh:, -1, 5].sum(axis=1) * nuh * 1.25
    grads.loc[:, r"$X_1$"] *= X1
    grads.loc[:, r"$X_2$"] *= X2
    grads.loc[:, r"$X_3$"] *= X3

    grads = grads.iloc[:, 3:-1]

    # Plot
    if args.zoom == 1:
        start, end = "1990", "2022"
    elif args.zoom == 2:
        start, end = "2021-10", "2022"
    elif args.zoom == 3:
        start, end = "2022-02-01", "2022-04-15"
    elif args.zoom == 4:
        start, end = "2022-02-22", "2022-03-10"

    # Metrics
    iok = qobs.notnull()
    qo, qs = qobs[iok][start:end], qsim[iok][start:end]
    perf_df = perf(qo, qs)
    logger.info(f"\n{perf_df}")


    # --------------------------------------------------
    #logger.info("Plot vs GR4J")
    #mosaic = [["S", "PR", "AE"], ["R", "UH", "Q"]]
    #ncols, nrows = len(mosaic[0]), len(mosaic)
    #figsize = (config.awidth * ncols, config.aheight * nrows)
    #plt.close("all")
    #fig = plt.figure(figsize=figsize, layout="constrained")
    #axs = fig.subplot_mosaic(mosaic, sharex=True)
    #for aname, ax in axs.items():
    #    if aname == "UH":
    #        v1 = grsim.loc[start:end, "Q1"] + grsim.loc[start:end, "Q9"]
    #    else:
    #        v1 = grsim.loc[start:end, aname]

    #    if aname == "S":
    #        v2 = mod.x[:, 1] * X1
    #    elif aname == "PR":
    #        v2 = -mod.fluxes[:, 1, 5]
    #    elif aname == "AE":
    #        v2 = -mod.fluxes[:, 0, 1] - mod.fluxes[:, 1, 1]
    #    elif aname == "R":
    #        v2 = mod.x[:, -1] * X3
    #    elif aname == "QR":
    #        v2 = -mod.fluxes[:, -1, 2]
    #    elif aname == "Q":
    #        v2 = -mod.fluxes[:, -1, 5]
    #    elif aname == "UH":
    #        v2 = -mod.fluxes[:, -2, 5]

    #    v2 = pd.Series(v2, index=grsim.index).loc[start:end]

    #    v1.plot(ax=ax, label="GR4J")
    #    v2.plot(ax=ax, label="Quas")
    #    ax.set(title=aname)

    #fp = script_paths.fimg / "gr4j_comparison.png"
    #fig.savefig(fp)

    # --------------------------------------------------
    logger.info("Plot grad ts")
    plt.close("all")

    nvars = grads.shape[1]
    nrows, ncols = nvars + 1, 1
    mosaic = [["flux", "."]] \
             + [[f"{vn}/ratio", f"{vn}/scatter"]
                for vn in grads.columns]
    ncols, nrows = len(mosaic[0]), len(mosaic)
    figsize = (config.awidth * ncols, config.aheight * nrows)
    fig = plt.figure(figsize=figsize,
                     layout="constrained")
    kw = dict(width_ratios=[3, 2])
    axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

    ax = axs["flux"]
    qobs[start:end].plot(ax=ax, label="Observed $q_o$", color="k", ls="--")
    qsim[start:end].plot(ax=ax, label="Simulated $Q_3$", lw=2)

    eps = qobs.mean() * 1e-2
    sqeps = math.sqrt(eps)
    fwd = lambda x: (np.sqrt(eps + np.abs(x)) - sqeps) * np.sign(x)
    inv = lambda y: np.sign(y) * ((np.abs(y) + sqeps)**2 - eps)
    u = np.linspace(-1, 1, 100)
    assert np.allclose(inv(fwd(u)), u)

    ax.set_yscale("function", functions=(fwd, inv))
    title = "(a) Streamflow timeseries"
    ax.set_title(title, va="top", ha="left",
                 x=0.02, y=0.94)
    ax.set(ylabel="Streamflow [mm.d$^{-1}$]")
    ax.legend(loc=3, ncol=2, framealpha=0.)

    # Plotting data
    res = qsim - qobs
    x = pd.DataFrame(mod.x, index=time)

    for iplot, (vn, dq) in enumerate(grads.items()):
        logger.info(f"Plot {iplot}", ntab=1)
        # Ts plot
        ax = axs[f"{vn}/ratio"]

        ratio = (res / -dq)
        ratio[start:end].plot(ax=ax)

        ax.set_yscale("asinh")
        putils.line(ax, 1, 0, 0, 0, "k-", lw=0.8)

        vtxt = re.sub("\\$", "", vn)
        title = f"({letters[2 * iplot + 1]}) "\
                + f"{vn} parameter error $\\frac{{Q_3 - q_o}}{{ {{{vtxt}}}\\ dQ_3/d{{{vtxt}}}}}$"
        t = ax.set_title(title, va="top", ha="left",
                         x=0.02, y=0.935, fontsize="x-large")
        t.set_bbox(dict(alpha=1.0, ec="none", fc="w"))

        ylim = -2e2, 2e2
        ax.set(ylim=ylim,
               ylabel="Parameter error [-]")

        # .. scatter plot
        ax = axs[f"{vn}/scatter"]

        if re.search("X_1", vn):
            ires = 1
            store_name = "Production store level"
        else:
            ires = x.shape[1] - 1
            store_name = "Routing store level"

        store = x.iloc[:, ires]
        ax.plot(store, ratio, ".", alpha=0.1)

        xx, yy = smooth_reg(store, ratio)
        ax.plot(xx, yy, "--", color="k", lw=2)

        s, e = store.index[[0, -1]]
        txt = f"Data from {s.year} to {e.year}"
        ax.text(0.98, 0.02, txt, va="bottom", ha="right",
                transform=ax.transAxes)

        title = f"({letters[2 * iplot + 2]})"
        ax.set_title(title, va="top", ha="left",
                     x=0.02, y=0.94)
        ax.set_yscale("asinh", linear_threshold=1e-3)
        putils.line(ax, 1, 0, 0, 0, "k-", lw=0.8)
        ax.set(ylim=ylim,
               xlabel=store_name,
               yticklabels=[])


    fp = script_paths.fimg / "gr4j_gradients_ts.png"
    fig.savefig(fp)

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
