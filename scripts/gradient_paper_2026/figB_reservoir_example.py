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

# Package to plot spatial data
from hydrodiy.io import csv, iutils

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

    names = ["t", "j", "niter", "t_start", "t_end", "time",
             "x_start", "x_end", "integ_x",
             "xi_start", "xi_end", "integ_xi",
             "q1", "q2", "q3",
             "Q1", "Q2", "Q3"]
    outputs = {n : [] for n in names}

    # Timeseries loop
    for t in range(nval):
        if debug:
            logger.info(f"Timestep {t}")
        # Inflow and demand potential fluxes
        qst1 = Qstars[t, 0]
        qst2 = Qstars[t, 1]
        qst3 = -gamma / timestep

        # Initialise timestep loop
        niter = 0
        t_start = 0
        t_end = 0
        Q1, Q2, Q3, Q4, Q5, Q6 = [0] * 6

        # interpolation band
        j = np.sum(x0 - alphas >= 0) - 1

        while t_end < timestep:
            mess = f"\nt={t:<2d} - n={niter:<2d} - j={j:<2d}\n" \
                    + f"\tt_sta={t_start:9.3e} " \
                    + f"x_stat={x_start:9.3e}\n"

            # Coefficients
            mu1j, mu2j, mu3j = coefs[:, j, 1]
            Mj = (qst1 * mu1j + qst2 * mu2j + qst3 * mu3j) / gamma

            sig1j, sig2j, sig3j = coefs[:, j, 0]
            Sj = (qst1 * sig1j + qst2 * sig2j + qst3 * sig3j) / gamma

            # Integrate forward
            funval = Mj + Sj * x_start
            x_end = forward(Mj, Sj, t_start, x_start, timestep)

            a0 = alphas[j]
            a1 = alphas[j + 1]

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
                    j = max(0, j - 1)
                else:
                    # .. above
                    x_end = a1
                    j = min(len(alphas) - 1, j + 1)

                # Finds the time where we hit the boundary of the band
                tau = inverse(Mj, Sj, x_start, x_end)
                if tau <= 0:
                    errmsg = "tau < 0"
                    #raise ValueError(errmsg)

                t_end = t_start + tau

            mess += f"\tt_end={t_end:9.3e} x_end={x_end:9.3e} j={j}\n"
            if debug:
                logger.info(mess)

            if isnull(x_end - x_start):
                errmsg = "x_end = x_start"
                if debug:
                    logger.error(errmsg)
                    import pdb; pdb.set_trace()
                raise ValueError(errmsg)

            # Integral of x between t_start and t_end
            tau = t_end - t_start
            integ_x = (x_end - x_start) / Sj - Mj / Sj * tau

            # Integral of fluxes between t_start and t_end
            if isnull(Sj):
                q1 = tau * (mu1j + sig1j * (x_start + Mj * tau / 2))
                q2 = tau * (mu2j + sig2j * (x_start + Mj * tau / 2))
                q3 = tau * (mu3j + sig3j * (x_start + Mj * tau / 2))
            else:
                Rj = Mj / Sj
                q1 = qst1 * ((mu1j - sig1j * Rj) * tau + sig1j / Sj * (x_end - x_start))
                q2 = qst2 * ((mu2j - sig2j * Rj) * tau + sig2j / Sj * (x_end - x_start))
                q3 = qst3 * ((mu3j - sig3j * Rj) * tau + sig3j / Sj * (x_end - x_start))

            Q1 += q1
            Q2 += q2
            Q3 += q3

            Q4 += q1 - qst1 * tau / timestep # Bypass inflow
            Q5 += q2 - qst2 * tau / timestep # Demand shortfall
            Q6 += q3 + q1 - qst1 * tau / timestep # Outflow

            # Gradient of coefficients
            dMj = np.array([-(qst1 * mu1j + qst2 * mu2j) / gamma**2,
                            qst1 * mu1j / gamma,
                            qst2 * mu2j / gamma,
                            mu3j / timestep])

            dSj = np.array([-(qst1 * sig1j + qst2 * sig2j) / gamma**2,
                            qst1 * sig1j / gamma,
                            qst2 * sig2j / gamma,
                            sig3j / timestep])


            # Sensitivity
            Wj = wfun(Sj, t_start, t_end)

            if isnull(Sj):
                xi_end = xi_start + dMj * tau

                integ_xi = dMj * tau**2 / 2
            else:
                dRj = (dMj - Rj * dSj) / Sj
                xi_end = Wj * (xi_start + dRj * (1 - 1. / Wj) + (x_start + Rj) * tau * dSj)

                integ_xi = (xi_end - xi_start - dMj * tau - dSj * integ_x) / Sj

            # Store
            outputs["t"].append(t)
            outputs["j"].append(j)
            outputs["niter"].append(niter)
            outputs["t_start"].append(t_start)
            outputs["t_end"].append(t_end)
            outputs["time"].append(t + t_end)

            outputs["x_start"].append(x_start)
            outputs["x_end"].append(x_end)
            outputs["integ_x"].append(integ_x)

            outputs["q1"].append(q1)
            outputs["q2"].append(q2)
            outputs["q3"].append(q3)
            outputs["Q1"].append(Q1)
            outputs["Q2"].append(Q2)
            outputs["Q3"].append(Q3)

            outputs["xi_start"].append(xi_start)
            outputs["xi_end"].append(xi_end)
            outputs["integ_xi"].append(integ_xi)

            # Loop for next band
            x_start = x_end
            xi_start = xi_end
            t_start = t_end
            niter += 1

            if niter > 10 and debug:
                import pdb; pdb.set_trace()

            if niter > len(alphas) + 2:
                raise ValueError("Too many iterations")

    return outputs


def process(config, script_paths, logger, data):
    logger.info(f"Configure model", nret=1)
    hclim = data.hydroclim
    time = hclim.index
    nval = len(hclim)
    area = data.area
    timestep = 1

    # Build quadratic model
    nalphas = 10
    funs = gradients.FLUX_FUNCTIONS["quadrouting"]
    alphas = gradients.get_alphas(nalphas)[0]
    #funs = [
    #    lambda x: np.ones_like(x),
    #    lambda x: np.zeros_like(x),
    #    lambda x: x**2
    #    ]
    #alphas = np.linspace(0, 1, 5)
    coefs = gradients.lin_coefficient_matrix(funs, alphas)

    # Set inputs
    Qstars = np.zeros((nval, 2))
    Qstars[:, 0] = hclim.Q

    # Set params
    gamma = 300

    # Run model timestep-by-timestep
    logger.info("Integrate details")
    x0 = 0.01
    debug = True
    outputs = integrate_details(alphas, gamma, Qstars, coefs, x0,
                                timestep=timestep,
                                logger=logger,
                                debug=debug)

    # Run model for complete timeseries
    niter, x, dx, fluxes, dfluxes = gradients.grad_reservoir(alphas, gamma,
                                                             Qstars, coefs,
                                                             x0=x0)

    plt.close("all")
    figsize = (config.awidth * 2, config.aheight)
    fig, axs = plt.subplots(ncols=2,
                           layout="constrained",
                           figsize=figsize)
    ax = axs[1]
    ax.plot(time, Qstars[:, 0], label="input")

    ax.plot(time, -fluxes[:, 2], label="C")

    t = np.array([time[0] + pd.Timedelta(days=d) for d in outputs["time"]])
    q = -np.array(outputs["Q3"])
    ax.plot(t, q, label="detail")

    #tax = ax.twinx()
    #tax.plot(time, niter, color="0.6")

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
    awidth = 6
    aheight = 5
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
