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
from pathlib import Path
from string import ascii_letters as letters

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
model_name = "GR"
siteid = "203024"

flux_names = ["eff_rain", "actual_et", "percol"]
flux_long_names = {
    "eff_rain": "Effective rainfall",
    "actual_et": "Actual ET",
    "percol": "Percolation"
    }

start_daily = "2010-01-01"
end_daily = "2022-12-31"

approx_long_names = {
    "precise": "Precise computation",
    "approx_midpt": "Approx. based on mid-point store",
    "approx_midptfun": "Approx. based on mid-point flux"
    }

# Plot config
colors = {
    "precise": "tab:blue",
    "approx_midpt": "tab:green",
    "approx_midptfun": "tab:orange"
    }

fdpi = 300

#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fout = froot / "outputs" / "supplementary_flux_computation"
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
daily = data_utils.get_data(siteid, "daily").loc[start_daily:end_daily]

rain = daily.loc[:, "RAINFALL[mm/day]"]
evap = daily.loc[:, "PET[mm/day]"]
climate = np.column_stack([rain, evap])

# Run models
# .. model setup
param = 10
timestep = 1. # time step in days
s0 = 0.5

# .. fluxes functions
fluxes, dfluxes = benchmarks.gr4jprod_fluxes_noscaling()

LOGGER.info("Run model")
X1 = param
time_index = daily.index
ones = np.ones(len(climate))
scalings = [np.maximum(rain-evap, 0)/X1, \
                        np.maximum(evap-rain, 0)/X1, ones]
scalings = scalings+[ones] if model_name.startswith("GRM") else scalings
scalings = np.column_stack(scalings)

# Numerical solver
niter, s1, sim = slow.numerical_model(fluxes, dfluxes,
                                      scalings, s0,
                                      timestep, method="Radau")
sim = param*np.abs(sim) / timestep

cns = [f"{fn}_precise" for fn in flux_names]
sim = pd.DataFrame(sim, columns=cns, index=rain.index)
sim.loc[:, "s"] = s1

# Get store at the beginning and end of timestep
s_end = sim.s
s_start = sim.s.shift(1)

# Compute approximations of fluxes
for ifx, fn in enumerate(flux_names):
    # Flux function
    fx = np.vectorize(fluxes[ifx])

    # Scalings
    sc = scalings[:, ifx]

    # Flux sign (flux in or out of the store)
    sgn = 1 if ifx == 0 else -1

    # mid-point store
    cbase = f"{flux_names[ifx]}"
    sim.loc[:, f"{cbase}_approx_midpt"] = X1*sgn*timestep*sc*fx((s_end+s_start)/2)

    # mid-point function
    sim.loc[:, f"{cbase}_approx_midptfun"] = X1*sgn*timestep*sc*(fx(s_end)+fx(s_start))/2

    # end-point
    #sim.loc[:, f"{cbase}_approx_endpt"] = X1*sgn*timestep*sc*fx(s_end)

LOGGER.info("Plot")
plt.close("all")

fig = plt.figure(figsize=(15, 12), layout="constrained")
mosaic = [[f"{fn}/ts", f"{fn}/scat"] for fn in flux_names]
kw = dict(width_ratios=[3, 1])
axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

for iax, (aname, ax) in enumerate(axs.items()):
    fxn, ptype = aname.split("/")
    df = sim.filter(regex=fxn, axis=1)
    df.columns = [re.sub(f"{fxn}_", "", cn)
                  for cn in df.columns]

    if ptype == "ts":
        xlabel = ""
        ylabel = "Flux [mm $day^{-1}$]"
        title = "Simulated times series"

        for cn in df.columns:
            lw = 2 if cn == "precise" else 1
            se = df.loc["2022-01":"2022-03", cn]
            lab = approx_long_names[cn]
            ax.plot(se, lw=lw, color=colors[cn], label=lab)
    else:
        xlabel = "Precise flux computation [mm $day^{-1}$]"
        ylabel = "Approximated flux [mm $day^{-1}$]"
        title = "Comparison approx. vs precise"

        x = df.filter(regex="precise", axis=1).squeeze()
        for cn in df.columns:
            if cn == "precise":
                continue

            lab = approx_long_names[cn]
            ax.plot(x, df.loc[:, cn], "o", color=colors[cn], label=lab)

    title = f"({letters[iax]}) {flux_long_names[fxn]}\n{title}"
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if ptype == "ts":
        ax.legend(loc=1)

fp = fout / f"flux_approximation.png"
fig.savefig(fp, dpi=fdpi)

LOGGER.completed()

