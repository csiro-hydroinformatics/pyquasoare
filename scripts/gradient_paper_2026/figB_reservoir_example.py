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

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils, violinplot

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

from pyquasoare import gradients
import c_pyquasoare_grad
from gradbench import data_reader

# Select backend
mpl.use("Agg")


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fimg = froot / "images" / "manuscript" / "reservoir_example"
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
    hydroclim = hydroclim.loc["2022-02-27":"2022-03-04"]

    hydroclim = hydroclim.rename(columns={
        "RAINFALL[mm_day-1]": "P",
        "PET[mm_day-1]": "PET",
        "STREAMFLOW[mm_day-1]": "Q"
        })
    hydroclim = hydroclim.loc[:, ["P", "PET", "Q"]]

    data = namedtuple("Data", ["hydroclim", "area"])(hydroclim, area)
    logger.info(".. done")

    return data


def wfun(S, t0, t):
    return math.exp(S * (t - t0))


def isnull(x):
    return abs(x) < 1e-20


def forward(M, S, t0, x0, t):
    if isnull(S):
        return x0 + M * (t - t0)
    else:
        R = M / S
        return -R + (x0 + R) * wfun(S, t0, t)


def inverse(M, S, x0, x):
    if isnull(S):
        tau = (x - x0) / M
    else:
        tau = 1. / S * math.log(abs((S * x + M) / (S * x0 + M)))

    return tau

def integrate_details(alphas, gamma, Qstars, coefs, x0,
                      timestep, logger, debug=False):
    nval = len(Qstars)

    # Initialise
    x_start = x0
    xi_start = np.zeros(4) # 4 variables = 1 capacity + 3 flux scalings

    scalars = ["t", "j", "niter", "t_start", "t_end", "time",
             "x_start", "x_end", "integ_x",
             "xi_start", "end_timestep"]
    vectors = ["xi_start", "xi_end", "integ_xi",
               "dfxG", "dfxH", "q", "Q"]
    outputs = {"scalars": []}
    for v in vectors:
        outputs[v] = []

    # interpolation band
    j = np.sum(x0 - alphas >= 0) - 1

    # Timeseries loop
    for t in range(nval):
        # Inflow and demand potential fluxes
        qst = np.array([Qstars[t, 0], Qstars[t, 1], -gamma / timestep])

        # Initialise timestep loop
        niter = 0
        t_start = 0
        t_end = 0
        Q = np.zeros(3)

        dfxG = np.zeros((4, 3))
        dfxH = np.zeros((4, 3))

        sc = {
            "t": t, "j": j, "niter": niter,
            "t_start": t_start, "t_end": t_end,
            "time": t + t_end,
            "x_start": x_start, "x_end": x_start,
            "integ_x": 0.,
            "end_timestep": 0
            }
        outputs["scalars"].append(sc)
        outputs["xi_start"].append(xi_start)
        outputs["xi_end"].append(xi_start)
        outputs["integ_xi"].append(xi_start * 0.)

        outputs["q"].append(np.zeros(3))
        outputs["Q"].append(Q.copy())
        outputs["dfxG"].append(dfxG.copy())
        outputs["dfxH"].append(dfxH.copy())

        # Quasoare loop
        while not isnull(timestep - t_end):
            # Coefficients
            musj = coefs[:, j, 1]
            Mj = (qst * musj / gamma).sum()

            sigsj = coefs[:, j, 0]
            Sj = (qst * sigsj / gamma).sum()

            a0 = alphas[j]
            a1 = alphas[j + 1]

            # Integrate forward
            funval = Mj + Sj * x_start
            x_end = forward(Mj, Sj, t_start, x_start, timestep)

            if x_start < a0 or x_start > a1:
                raise ValueError("Out of band")

            if np.isnan(x_end):
                raise ValueError("x_end is nan")

            elif x_end >= a0 and x_end <= a1:
                # We stay within the same interpolation band
                # for the rest of the timestep
                t_end = timestep
            else:
                # We go out of the current band
                if funval < 0:
                    # .. below
                    x_end = a0
                    j -= 1
                else:
                    # .. above
                    x_end = a1
                    j += 1

                if j < 0 or j >= len(alphas) - 1:
                    raise ValueError("x going out of range")

                # Finds the time where we hit the boundary of the band
                tau = inverse(Mj, Sj, x_start, x_end)
                if tau <= 0:
                    raise ValueError("tau < 0")

                t_end = t_start + tau

            if isnull(x_end - x_start):
                raise ValueError("x_end = x_start")

            # Integral of x and f(x) between t_start and t_end
            tau = t_end - t_start
            integ_x = (x_end - x_start) / Sj - Mj / Sj * tau
            integ_fx = sigsj * integ_x + musj * tau

            # Integral of fluxes between t_start and t_end
            q = qst / timestep * integ_fx
            Q += q

            # Gradient of coefficients
            dMj = np.array([-(qst[0] * musj[0] + qst[1] * musj[1]) / gamma**2,
                            qst[0] * musj[0] / gamma,
                            qst[1] * musj[1] / gamma,
                            -musj[2] / timestep])

            dSj = np.array([-(qst[0] * sigsj[0] + qst[1] * sigsj[1]) / gamma**2,
                            qst[0] * sigsj[0] / gamma,
                            qst[1] * sigsj[1] / gamma,
                            -sigsj[2] / timestep])

            # Sensitivity
            Wj = wfun(Sj, t_start, t_end)

            if isnull(Sj):
                xi_end = xi_start + dMj * tau

                integ_xi = dMj * tau**2 / 2
            else:
                Rj = Mj / Sj
                dRj = (dMj - Rj * dSj) / Sj
                xi_end = Wj * xi_start + dRj * (Wj - 1.)\
                    + (x_start + Rj) * tau * dSj * Wj

                integ_xi = (xi_end - xi_start - dMj * tau - dSj * integ_x) / Sj

            # Flux derivatives
            # .. first term (G)
            g = np.zeros((4, 3))
            g[0, 2] = -integ_fx[2] / timestep**2
            dfxG += g

            # .. second term (H)
            dm = np.zeros((4, 3))
            ds = np.zeros((4, 3))
            k = np.arange(3)
            dm[k + 1, k] = musj
            ds[k + 1, k] = sigsj
            dfxH += qst[None, :] / timestep * (dm * tau + ds * integ_x
                                               + sigsj[None, :] * integ_xi[:, None])

            # Store
            sc = {
                "t": t, "j": j, "niter": niter,
                "t_start": t_start, "t_end": t_end,
                "time": t + t_end / timestep,
                "x_start": x_start, "x_end": x_end,
                "integ_x": integ_x,
                "end_timestep": int(isnull(timestep - t_end))
                }
            outputs["scalars"].append(sc)
            outputs["q"].append(q)
            outputs["Q"].append(Q.copy())
            outputs["xi_start"].append(xi_start)
            outputs["xi_end"].append(xi_end)
            outputs["integ_xi"].append(integ_xi)

            outputs["dfxG"].append(dfxG.copy())
            outputs["dfxH"].append(dfxH.copy())

            # Loop for next band
            x_start = x_end
            xi_start = xi_end
            t_start = t_end
            niter += 1

            if niter > len(alphas) + 2:
                raise ValueError("Too many iterations")

    outputs["scalars"] = pd.DataFrame(outputs["scalars"])
    for v in vectors:
        if re.search("dfx", v):
            outputs[v] = np.array(outputs[v])
        else:
            outputs[v] = np.vstack(outputs[v])

    return outputs


def plot_timeseries(ax, time, values, end_timestep, color,
                    continuous=False):
    nval = len(time) - 1
    for i in range(nval):
        t, v = time[i], values[i]
        tn, vn = time[i + 1], values[i + 1]
        if end_timestep[i] == 1:
            continue

        if continuous:
            ax.plot([t, tn], [v, vn], "-", color=color)
        else:
            ax.plot([t, t], [v, vn], "-", color=color)
            ax.plot([t, tn], [vn, vn], "-", color=color)

        ax.plot(t, v, ".", color=color)
        ax.plot(tn, vn, "o", mec=color, mfc="w", zorder=10)

    ax.plot([], [], "o-", mec=color, mfc="w", color=color,
            label="Intermediary results")

    idx = end_timestep == 1
    ms = 8
    ax.plot(time[idx], values[idx], "o", color=color,
            ms=ms, label="End of timestep",
            zorder=11)


def format_time_xaxis(ax, time, label=True):
    ax.xaxis.set_major_locator(mdates.DayLocator())
    fmt = mdates.DateFormatter("")
    ax.xaxis.set_major_formatter(fmt)
    ax.grid(axis="x")

    if label:
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
        fmt = mdates.DateFormatter("%d %b\n%y")
        ax.xaxis.set_minor_formatter(fmt)
        ax.tick_params(axis='x', which='minor', tick1On=False, tick2On=False)


def process(config, script_paths, logger, data):
    logger.info(f"Configure model", nret=1)
    hclim = data.hydroclim
    nsubsteps = 1
    timestep = 1. / nsubsteps
    delta = pd.Timedelta(days=timestep)

    shift = np.tile(delta * np.arange(nsubsteps),
                    (len(hclim)))
    time = np.repeat(hclim.index, nsubsteps) + shift

    nval = len(hclim)
    area = data.area

    # Build quadratic model
    funs = gradients.FLUX_FUNCTIONS["quadrouting"]
    #nalphas = 15
    #alphas = gradients.get_alphas(nalphas)[0]
    alphas = 0.2 * np.arange(6)
    nalphas = len(alphas)
    coefs = gradients.lin_coefficient_matrix(funs, alphas)

    # Set inputs
    Qstars = np.zeros((nsubsteps * nval, 2))
    Qstars[:, 0] = np.repeat(hclim.Q, nsubsteps)
    #Qstars[:, 1] = -5. # Demand

    # Set params
    gamma = 200

    # Run model timestep-by-timestep
    logger.info("Integrate details")
    x0 = 0.1
    debug = False
    outputs = integrate_details(alphas, gamma, Qstars, coefs, x0,
                                timestep=timestep,
                                logger=logger,
                                debug=debug)
    sc = outputs["scalars"]
    sc.loc[:, "date"] = time[0] + (sc.time - 1) * delta

    # Run C code to check results
    niter, x, dx, fluxes, dfluxes = gradients.grad_reservoir(alphas, gamma,
                                                             Qstars, coefs,
                                                             x0=x0,
                                                             timestep=timestep)

    iend = sc.end_timestep == 1
    assert np.allclose(sc.loc[iend, "x_end"], x)
    assert np.allclose(outputs["Q"][iend], fluxes[:, :3])

    logger.info("print table")
    import pdb; pdb.set_trace()


    logger.info("plot")

    plt.close("all")
    ncols = 3
    nrows = 2
    figsize = (config.awidth * ncols, config.aheight * nrows)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
                            layout="constrained",
                            figsize=figsize)

    # QuaSoARe approximation
    ax = axs[0, 0]
    x = np.linspace(0, 1, 100)
    dx = -x**2
    ax.plot(dx, x, "-", color="0.4", label="True")

    dxa = -alphas**2
    ax.plot(dxa, alphas, "s",
            color="0.6", label="Interpolation nodes")
    dxi = np.interp(x, alphas, dxa)
    col = "tab:blue"
    ax.plot(dxi, x, "-", color=col,
            label="Linear interpolation")

    title = "(a) Linear approximation of discharge flux function"
    ax.set(title=title, yticks=alphas)
    ax.legend()

    # reservoir level timeseries
    ax = axs[1, 0]
    title = "(d) Potential Inflow $Q^*_1$"
    ax.step(time, Qstars[:, 0], where="post")
    ax.set(title=title)
    format_time_xaxis(ax, time)

    # reservoir level timeseries
    ax = axs[0, 1]
    col = "tab:blue"
    plot_timeseries(ax, sc.date, sc.x_end, sc.end_timestep, col,
                    True)
    title = "(b) Reservoir filling level ($x$)"
    ax.set(title=title, yticks=alphas)
    format_time_xaxis(ax, time, False)
    ax.sharey(axs[0, 0])
    ax.legend(loc=2, fontsize="large")

    # derivative of reservoir level timeseries
    ax = axs[0, 2]
    dx = outputs["xi_end"][:, 0]
    col = "tab:blue"
    plot_timeseries(ax, sc.date, dx, sc.end_timestep, col,
                    True)
    title = "(c) Gradient of reservoir filling level ($dx/d\\Gamma$)"
    ax.set(title=title)
    format_time_xaxis(ax, time, False)

    # Flux timeseries
    ax = axs[1, 1]
    q = outputs["Q"][:, 2]
    col = "tab:blue"
    plot_timeseries(ax, sc.date, q, sc.end_timestep, col)
    title = "(e) Discharge flux ($Q_3$)"
    ax.set(title=title)
    format_time_xaxis(ax, time)

    # Flux derivative timeseries
    ax = axs[1, 2]
    dq = (outputs["dfxG"] + outputs["dfxH"])[:, 0, 2]
    plot_timeseries(ax, sc.date, dq, sc.end_timestep, col)
    title = "(f) Gradient of discharge ($dQ_3/d\\Gamma$)"
    ax.set(title=title)
    format_time_xaxis(ax, time)

    # Save file
    fp = script_paths.fimg / \
            f"figB_resevoir_example.{config.imgext}"
    fig.savefig(fp, dpi=config.fdpi, transparent=config.ftransparent)
    #putils.blackwhite(fp)


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
    stationid = args.stationid
    awidth = 5
    aheight = 4
    fdpi = 300
    imgext = "png"
    ftransparent = False

    Config = namedtuple("Config", ["awidth",
                                   "aheight", "fdpi",
                                   "ftransparent", "imgext",
                                   "timestep", "stationid"])
    config = Config(awidth, aheight, fdpi,
                    ftransparent, imgext,
                    timestep, stationid)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
