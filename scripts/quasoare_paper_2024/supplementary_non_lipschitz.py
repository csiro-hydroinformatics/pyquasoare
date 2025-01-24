#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-01-23 Thu 10:46 PM
## Comment : Run a model with flux functions that are not
##           Lipschitz continuous
##
## ------------------------------

import sys, os, re, json, math
import argparse
import inspect
from pathlib import Path
import time
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.dates as mdates

from hydrodiy.io import csv, iutils

from pyquasoare import approx, steady, benchmarks, models, slow

import hdf5_utils
import data_utils

#----------------------------------------------------------------------
# @Config
#----------------------------------------------------------------------
# Get model name and siteid
model_name = "SQ"
siteid = "203024"

# List of ODE ode_methods
ode_methods = ["radau", "c_quasoare_3", "c_quasoare_10"]

start_daily = "2010-01-01"
end_daily = "2022-12-31"

start_hourly = "2022-02-01"
end_hourly = "2022-04-10"

nparams = 1

# Plot config
col_ana = "grey"
ls_ana = "--"
lw_ana = 3

col_qua = "tab:blue"
ls_qua = "-"
lw_qua = 3

col_rad = "tab:orange"
ls_rad = "-"
lw_rad = 6


#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fout = froot / "outputs" / "supplementary_non_lipschitz"
fout.mkdir(exist_ok=True, parents=True)

#----------------------------------------------------------------------
# @Logging
#----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# @Process
#----------------------------------------------------------------------
LOGGER.info("Load data")

# Prepare data for models
hourly = data_utils.get_data(siteid, "hourly").loc[start_hourly:end_hourly]
inflows = hourly.loc[:, "STREAMFLOW_UP[m3/sec]"].interpolate()
q0 = inflows.quantile(0.9)

# Run models
# .. model setup
timestep = 3600. # time step in seconds
# store q0 for 6 hours
param = q0*timestep*6

# routing exponent
nu = 0.5
s0 = 1e-3

# .. fluxes functions
fluxes, dfluxes = benchmarks.nonlinrouting_fluxes_noscaling(nu)

# Run model
theta = param
time_index = hourly.index
nval = len(inflows)
scalings = np.column_stack([inflows/theta, \
                            q0/theta*np.ones(nval)])

# Loop over ODE ode_method
simall = []
flux_approx = {}
nodes = {}

for ode_method in ode_methods:
    LOGGER.info(f"{ode_method} - start")
    ode_text = re.sub("c_", "", ode_method)

    if ode_method == "radau":
        # Numerical solver
        tstart = time.time()
        niter, s1, sim = slow.numerical_model(fluxes, dfluxes,
                                              scalings, s0,
                                              timestep, method="Radau")
        runtime = (time.time()-tstart)*1e3
        sim = param*np.abs(sim) / timestep
        s1_min = np.nanmin(s1)
        s1_max = np.nanmax(s1)
        alpha_min, alpha_max = np.nan, np.nan

    elif re.search("quasoare", ode_method):
        # first go at alphas
        opt = 0 if re.search("lin", ode_method) else 1
        alphas = np.linspace(0, 5., 500)
        amat, bmat, cmat, cst = approx.quad_coefficient_matrix(fluxes,
                                                 alphas,
                                                 approx_opt=opt)
        stdy = steady.quad_steady_scalings(alphas, scalings,
                                           amat, bmat, cmat)

        # second go at alphas
        nalphas = int(re.sub(".*_", "", ode_method))
        alpha_min = 0.
        alpha_max = np.nanmax(stdy)
        alphas = np.linspace(alpha_min, alpha_max, nalphas)
        amat, bmat, cmat, cst = approx.quad_coefficient_matrix(fluxes,
                                                               alphas,
                                                               approx_opt=opt)
        nfluxes = amat.shape[1]
        mean_scalings = scalings.mean(axis=0)*param
        no = {f"flux{ifx+1}": pd.Series([fluxes[ifx](a)*mean_scalings[ifx] for a in alphas],
                                      index=alphas) for ifx in range(nfluxes)}
        nodes[ode_text] = pd.DataFrame(no)

        # Store flux approximation
        xx = np.linspace(alpha_min, alpha_max, 500)
        fx = approx.quad_fun_from_matrix(alphas, amat, bmat, cmat, xx)

        # .. multiply by average scaling times store capacity
        #    to obtain sensible values
        fx *= mean_scalings[None, :]

        dfx = np.empty(fx.shape[0], dtype=hdf5_utils.FLUXES_DTYPE)
        dfx["s"] = xx
        for iflux in range(4):
            if iflux < nfluxes:
                f = np.array([fluxes[iflux](x)*mean_scalings[iflux] for x in xx])
                dfx[f"flux{iflux+1}_true"] = f
                dfx[f"flux{iflux+1}_approx"] = fx[:, iflux]
            else:
                dfx[f"flux{iflux+1}_true"] = np.nan
                dfx[f"flux{iflux+1}_approx"] = np.nan

        dfx = pd.DataFrame(dfx)
        flux_approx[ode_text] = dfx

        # Run model
        quad_model = models.quad_model if ode_method.startswith("c")\
                                else slow.quad_model
        tstart = time.time()
        try:
            niter, s1, sim = quad_model(alphas, scalings, \
                                amat, bmat, cmat, s0, timestep)
            runtime = (time.time()-tstart)*1e3
            sim = param*np.abs(sim)/timestep
            s1_min = np.nanmin(s1)
            s1_max = np.nanmax(s1)
        except Exception as err:
            LOGGER.error(f"Error in {ode_method} run")
            LOGGER.error(str(err))
            sim = None

    if sim is not None:
        otxt = re.sub("c_", "", ode_method)
        cols = [f"qin", f"{otxt}_qout"]
        s = pd.DataFrame(sim, columns=cols, index=inflows.index)
        s.loc[:, f"{otxt}_s"] = s1
        cc = s.columns if ode_method == "radau" else \
            [cn for cn in s.columns if cn != "qin"]
        simall.append(s.loc[:, cc])

simall = pd.concat(simall, axis=1)

# Plot
plt.close("all")

fig = plt.figure(figsize=(10, 5), layout="constrained")
mosaic = [[om for om in flux_approx]]
axs = fig.subplot_mosaic(mosaic)

# .. fluxes
for iom, (om, dfx) in enumerate(flux_approx.items()):
    ax = axs[om]

    # True
    s = dfx.s/theta
    y = dfx.flux2_true
    ax.plot(s, y, color=col_ana, lw=lw_ana,
            linestyle=ls_ana)

    # Approx
    s = dfx.s/theta
    y = dfx.flux2_approx
    ax.plot(s, y, color=col_qua, lw=lw_qua,
            linestyle=ls_qua)

    # Nodes
    no = nodes[om]
    s = no.index/theta
    y = no.flux2
    ax.plot(s, y, "o", color="tab:red")

    # Decorate
    unit = "m$^3$ s$^{-1}$"
    n = re.sub(".*_", "", om)
    ax.set(xlabel="Store filling level $S/{\\theta}$ [-]",
           ylabel=f"Instantaneous flux [{unit}]",
           title=f"Outflow - QuaSoARe {n}")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

fp = fout / f"flux_approximation.png"
fig.savefig(fp)

# .. time series
fig = plt.figure(figsize=(10, 5), layout="constrained")
mosaic = [["s"], ["qout"]]
axs = fig.subplot_mosaic(mosaic)
start = "2022-02-20"
end = "2022-03-10"

for varn in ["s", "qout"]:
    ax = axs[varn]
    for iom, om in enumerate(ode_methods):
        omt = re.sub("c_", "", om)
        se = simall.loc[start:end, f"{omt}_{varn}"]
        if varn == "s":
            se /= theta

        col = col_qua if re.search("quas", om) else col_rad
        lw = 1 if om == ode_methods[1] else 2
        lab = f"{omt}"
        ax.plot(se, color=col, lw=lw, label=lab)

        #if varn == "qout" and iom == 0:
        #    ax.plot(inflows.loc[start:end], "k--", label="inflows", lw=1)

    ax.legend(loc=2)

    unit = "-" if varn == "s" else "m$^3$ s$^{-1}$"
    ylab = "Streamflow" if varn=="qout" else "Store filling level"
    ylab = f"{ylab} [{unit}]"
    ax.set(ylabel=ylab)

    fmt = mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    ax.xaxis.set_major_formatter(fmt)

    if varn == "s":
        def fmt(x, pos):
            base, expon = re.split("e", f"{x:1.1e}")
            if set(expon[1:]) == {"0"}:
                lab = base
            else:
                expon = expon[0] + re.sub("^0+", "", expon[1:])
                expon = f"$10^{{{expon}}}$"
                lab = base + r"$\times$" + expon
            return lab

        ax.yaxis.set_major_formatter(fmt)

fp = fout / f"simuilations.png"
fig.savefig(fp)

LOGGER.completed()

