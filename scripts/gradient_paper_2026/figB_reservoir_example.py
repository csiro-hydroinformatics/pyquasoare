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
    #hydroclim = hydroclim.loc["2022-02-27":"2022-03-04"]
    hydroclim = hydroclim.loc["2022-02-27":"2022-02-28"]

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
             "xi_start"
             "q1", "q2", "q3",
             "Q1", "Q2", "Q3"]
    vectors = ["xi_start", "xi_end", "integ_xi",
               "dfxH1", "dfxH2", "dfxH3"]
    outputs = {"scalars": []}
    for v in vectors:
        outputs[v] = []

    # interpolation band
    j = np.sum(x0 - alphas >= 0) - 1

    # Timeseries loop
    for t in range(nval):
        # Inflow and demand potential fluxes
        qst1 = Qstars[t, 0]
        qst2 = Qstars[t, 1]
        qst3 = -gamma / timestep

        # Initialise timestep loop
        niter = 0
        t_start = 0
        t_end = 0
        Q1, Q2, Q3 = [0] * 3
        dfxH1 = np.zeros(4)
        dfxH2 = np.zeros(4)
        dfxH3 = np.zeros(4)

        sc = {
            "t": t, "j": j, "niter": niter,
            "t_start": t_start, "t_end": t_end,
            "time": t + t_end,
            "x_start": x_start, "x_end": x_start,
            "integ_x": 0.,
            "q1": 0., "q2": 0., "q3": 0.,
            "Q1": Q1, "Q2": Q2, "Q3": Q3
            }
        outputs["scalars"].append(sc)
        outputs["xi_start"].append(xi_start)
        outputs["xi_end"].append(xi_start)
        outputs["integ_xi"].append(xi_start * 0.)

        outputs["dfxH1"].append(dfxH1.copy())
        outputs["dfxH2"].append(dfxH2.copy())
        outputs["dfxH3"].append(dfxH3.copy())

        # Quasoare loop
        while not isnull(timestep - t_end):
            # Coefficients
            mu1j, mu2j, mu3j = coefs[:, j, 1]
            Mj = (qst1 * mu1j + qst2 * mu2j + qst3 * mu3j) / gamma

            sig1j, sig2j, sig3j = coefs[:, j, 0]
            Sj = (qst1 * sig1j + qst2 * sig2j + qst3 * sig3j) / gamma

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

            # Integral of x between t_start and t_end
            tau = t_end - t_start
            integ_x = (x_end - x_start) / Sj - Mj / Sj * tau

            # Integral of fluxes between t_start and t_end
            if isnull(Sj):
                q1 = tau * (mu1j + sig1j * (x_start + Mj * tau / 2)) / timestep
                q2 = tau * (mu2j + sig2j * (x_start + Mj * tau / 2)) / timestep
                q3 = tau * (mu3j + sig3j * (x_start + Mj * tau / 2)) / timestep
            else:
                Rj = Mj / Sj
                q1 = qst1 / timestep * ((mu1j - sig1j * Rj) * tau + sig1j / Sj * (x_end - x_start))
                q2 = qst2 / timestep * ((mu2j - sig2j * Rj) * tau + sig2j / Sj * (x_end - x_start))
                q3 = qst3 / timestep * ((mu3j - sig3j * Rj) * tau + sig3j / Sj * (x_end - x_start))

            Q1 += q1
            Q2 += q2
            Q3 += q3

            # Gradient of coefficients
            dMj = np.array([-(qst1 * mu1j + qst2 * mu2j) / gamma**2,
                            qst1 * mu1j / gamma,
                            qst2 * mu2j / gamma,
                            -mu3j / timestep])

            dSj = np.array([-(qst1 * sig1j + qst2 * sig2j) / gamma**2,
                            qst1 * sig1j / gamma,
                            qst2 * sig2j / gamma,
                            -sig3j / timestep])

            # Sensitivity
            Wj = wfun(Sj, t_start, t_end)

            if isnull(Sj):
                xi_end = xi_start + dMj * tau

                integ_xi = dMj * tau**2 / 2
            else:
                dRj = (dMj - Rj * dSj) / Sj
                xi_end = Wj * xi_start + dRj * (Wj - 1.)\
                    + (x_start + Rj) * tau * dSj * Wj

                integ_xi = (xi_end - xi_start - dMj * tau - dSj * integ_x) / Sj

            # Flux derivatives
            # .. Qstars are constants = derivative is null -> G = 0
            # .. only H term is computed
            dmu1 = np.array([0., mu1j, 0., 0.])
            dsig1 = np.array([0., sig1j, 0., 0.])
            dfxH1 += qst1 / timestep * (dmu1 * tau + dsig1 * integ_x + sig1j * integ_xi)

            dmu2 = np.array([0., 0., mu2j, 0.])
            dsig2 = np.array([0., 0., sig2j, 0.])
            dfxH2 += qst2 / timestep * (dmu2 * tau + dsig2 * integ_x + sig2j * integ_xi)

            dmu3 = np.array([0., 0., 0., mu3j])
            dsig3 = np.array([0., 0., 0., sig3j])
            dfxH3 += qst3 / timestep * (dmu3 * tau + dsig3 * integ_x + sig3j * integ_xi)

            # Store
            sc = {
                "t": t, "j": j, "niter": niter,
                "t_start": t_start, "t_end": t_end,
                "time": t + t_end / timestep,
                "x_start": x_start, "x_end": x_end,
                "integ_x": integ_x,
                "q1": q1, "q2": q2, "q3": q3,
                "Q1": Q1, "Q2": Q2, "Q3": Q3
                }
            outputs["scalars"].append(sc)
            outputs["xi_start"].append(xi_start)
            outputs["xi_end"].append(xi_end)
            outputs["integ_xi"].append(integ_xi)

            outputs["dfxH1"].append(dfxH1.copy())
            outputs["dfxH2"].append(dfxH2.copy())
            outputs["dfxH3"].append(dfxH3.copy())

            # Loop for next band
            x_start = x_end
            xi_start = xi_end
            t_start = t_end
            niter += 1

            if niter > len(alphas) + 2:
                raise ValueError("Too many iterations")

    outputs["scalars"] = pd.DataFrame(outputs["scalars"])
    for v in vectors:
        outputs[v] = np.vstack(outputs[v])

    return outputs


def process(config, script_paths, logger, data):
    logger.info(f"Configure model", nret=1)
    hclim = data.hydroclim
    nsubsteps = 4
    timestep = 1. / nsubsteps
    delta = pd.Timedelta(days=timestep)

    shift = np.tile(delta * np.arange(nsubsteps),
                    (len(hclim)))
    time = np.repeat(hclim.index, nsubsteps) + shift

    nval = len(hclim)
    area = data.area

    # Build quadratic model
    nalphas = 40
    funs = gradients.FLUX_FUNCTIONS["quadrouting"]
    alphas = gradients.get_alphas(nalphas)[0]
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

    # Run model for complete timeseries
    niter, x, dx, fluxes, dfluxes = gradients.grad_reservoir(alphas, gamma,
                                                             Qstars, coefs,
                                                             x0=x0,
                                                             timestep=timestep)
    # Numerical diff 2
    eps = 1e-6
    dx_n = dx.copy()
    dfluxes_n = dfluxes.copy()
    for iv in range(4):
        gamma_l = gamma - eps if iv == 0 else gamma
        gamma_h = gamma + eps if iv == 0 else gamma
        coefs_l = coefs.copy()
        coefs_h = coefs.copy()
        if iv >= 1:
            coefs_l[iv - 1] *= (1 - eps)
            coefs_h[iv - 1] *= (1 + eps)

        _, x_l, _, fluxes_l, _ = gradients.grad_reservoir(alphas, gamma_l,
                                                          Qstars, coefs_l,
                                                          x0=x0,
                                                          timestep=timestep)
        _, x_h, _, fluxes_h, _ = gradients.grad_reservoir(alphas, gamma_h,
                                                          Qstars, coefs_h,
                                                          x0=x0,
                                                          timestep=timestep)
        dx_n[:, iv] = (x_h - x_l) / 2 / eps
        dfluxes_n[:, iv, :] = (fluxes_h - fluxes_l) / 2 / eps

    plt.close("all")
    ncols = 5
    nrows = 3
    figsize = (config.awidth * ncols, config.aheight * nrows)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
                            layout="constrained",
                            figsize=figsize,
                            sharex=True)

    ax = axs[0, 0]
    ax.plot(time, x, "s-")
    ax.plot(sc.date, sc.x_end, ".-")
    ax.set(title="x")
    ax.grid(axis="x")

    v = outputs["xi_end"]
    for iv in range(4):
        ax = axs[0, 1 + iv]
        nu = gamma if iv == 0 else 1.
        ax.plot(time, nu * dx_n[:, iv], "-", color="0.5", lw=5,
                alpha=0.5)
        ax.plot(time, nu * dx[:, iv], "s-")
        ax.plot(sc.date, nu * v[:, iv], ".-")

        vtxt = "$\eta_\Gamma$" if iv == 0 else f"$\eta_{{{iv + 1}}}^f$"
        ax.set(title=f"diff x / {vtxt}")
        ax.grid(axis="x")

    ax = axs[1, 0]
    ax.plot(time, Qstars[:, 0], "ok--", label="input")
    tax = ax.twinx()
    tax.plot(time, fluxes[:, 2], "s-", label="Quas-C")
    q = sc.Q3.values
    tax.plot(sc.date, q, ".-", label="Quas-Py")
    ax.set(title="Q3")
    ax.legend()
    ax.grid(axis="x")

    #ax.plot(time, -fluxes[:, 1], "s-", label="C")
    #q = -sc.Q2.values
    #ax.plot(sc.date, q, ".-", label="detail")
    #ax.set(title="Q2")

    dfx = outputs["dfxH3"]
    for iv in range(4):
        ax = axs[1, 1 + iv]
        nu = gamma if iv == 0 else 1.

        ax.plot(time, nu * dfluxes_n[:, iv, 2], "-", color="0.5", lw=5,
                alpha=0.5, label="Numerical")
        ax.plot(time, nu * dfluxes[:, iv, 2], "s-", label="Quas-C")
        ax.plot(sc.date, nu * dfx[:, iv], ".-", label="Quas-Py")

        vtxt = "$\eta_\Gamma$" if iv == 0 else f"$\eta_{{{iv + 1}}}^f$"
        ax.set(title=f"diff Q3 / {vtxt}")

        putils.line(ax, 1, 0, 0, 0, "k-", lw=0.5)

        #ax.plot(time, nu * dfluxes_n[:, iv, 1], "-", color="0.5", lw=5,
        #        alpha=0.5, label="Numerical")
        #ax.plot(time, nu * dfluxes[:, iv, 1], "s-", label="Quas-C")
        #ax.plot(sc.date, nu * dfx[:, iv], ".-", label="Quas-Py")
        #ax.set(title=f"diff Q2 / v{iv + 1}")

        ax.grid(axis="x")
        if iv == 0:
            ax.legend()

    integ_dx = outputs["integ_xi"]
    for iv in range(4):
        ax = axs[2, 1 + iv]
        ax.plot(sc.date, integ_dx[:, iv], ".-", color="tab:orange", label="Quas-Py")
        ax.grid(axis="x")

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
