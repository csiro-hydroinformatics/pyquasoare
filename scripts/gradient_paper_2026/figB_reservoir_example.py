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


def forward(M, S, t0, x0, t):
    R = M / S
    return -R + (x0 + R) * wfun(S, t0, t)


def inverse(M, S, x0, x):
    return 1. / S * math.log(abs((S * x + M) / (S * x0 + M)));


def integrate_details(alphas, gamma, Qstars, coefs, x0, timestep):
    nval = len(Qstars)

    # Initialise
    x_start = x0
    xi_start = np.zeros(4) # 4 variables = 1 capacity + 3 flux scalings

    # Timeseries loop
    for t in range(nval):
        # Inflow and demand potential fluxes
        qst1 = Qstars[t, 0]
        qst2 = Qstars[t, 1]
        qst3 = -gamma / timestep

        # Initialise timestep loop
        t_start = 0
        t_end = 0
        Q1, Q2, Q3, Q4, Q5, Q6 = [0] * 6

        while t_end < timestep:
            # interpolation band
            j = np.sum(x0 - alphas >= 0) - 1

            # Coefficients
            mu1j, mu2j, mu3j = coefs[:, j, 0]
            Mj = (qst1 * mu1j + qst2 * mu2j + qst3 * mu3j) / gamma

            sig1j, sig2j, sig3j = coefs[:, j, 1]
            Sj = (qst1 * sig1j + qst2 * sig2j + qst3 * sig3j) / gamma

            # Integrate forward
            x_end = forward(Mj, Sj, t_start, x_start, timestep)
            funval = Mj + Sj * x_start
            if x_end >= alphas[j] and x_end <= alphas[j+1]:
                # We stay within the same interpolation band
                # for the rest of the timestep
                t_end = timestep
            else:
                # We go out of the current band
                if funval < 0:
                    j = max(0, j - 1)
                else:
                    j = min(len(alphas) - 1, j + 1)

                # Finds the time where we hit the boundary of the band
                x_end = alphas[j]
                t_end = t_start + inverse(Mj, Sj, x_start, x_end)

            # Integral of x between t_start and t_end
            tau = t_end - t_start
            integ_x = (x_end - x_start) / Sj - Mj / Sj * tau

            # Integral of fluxes between t_start and t_end
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

            dRj = (dMj - Rj * dSj) / Sj

            # Sensitivity
            Wj = wfun(Sj, t_start, t_end)
            xi_end = Wj * (xi_start + dRj * (1 - 1. / Wj) + (x_start + Rj) * tau * dSj)


            # Loop for next band
            x_start = x_end
            xi_start = xi_end
            t_start = t_end
            import pdb; pdb.set_trace()








def process(config, script_paths, logger, data):
    logger.info(f"Start plotting", nret=1)
    hclim = data.hydroclim
    time = hclim.index
    nval = len(hclim)
    area = data.area
    timestep = 1

    # Build quadratic model
    nalphas = 10
    funs = gradients.FLUX_FUNCTIONS["quadrouting"]
    alphas = gradients.get_alphas(nalphas)[0]
    coefs = gradients.lin_coefficient_matrix(funs, alphas)

    # Set inputs
    Qstars = np.zeros((nval, 2))
    Qstars[:, 0] = hclim.Q

    # Set params
    gamma = 200

    # Run model timestep-by-timestep
    x0 = 1e-5
    integrate_details(alphas, gamma, Qstars, coefs, x0, timestep=timestep)

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
    ax.plot(time, Qstars[:, 0])

    ax.plot(time, -fluxes[:, 2])

    tax = ax.twinx()
    tax.plot(time, niter, color="0.6")

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
