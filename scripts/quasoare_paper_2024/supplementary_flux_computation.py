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
from hydrodiy.plot import putils

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
    "s": "Store filling level",
    "eff_rain": "Infiltrated rain",
    "actual_et": "Actual ET",
    "percol": "Percolation"
    }

start_daily = "2010-01-01"
end_daily = "2022-12-31"

approx_long_names = {
    "radau": "Radau",
    "approx_midpt": "Approx - mid-point store",
    "approx_midptfun": "Approx - mid-point flux"
    }

# Plot config
colors = {
    "radau": "tab:orange",
    "approx_midpt": "tab:green",
    "approx_midptfun": "tab:purple"
    }

fdpi = 300

putils.set_mpl(font_size=13)

#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

fimg = froot / "images" / "supplementary_flux_computation"
fimg.mkdir(exist_ok=True, parents=True)

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
param = 50
timestep = 1. # time step in days
s0 = 0.5

# .. fluxes functions
fluxes, dfluxes = benchmarks.gr4jprod_fluxes_noscaling()

X1 = param
time_index = daily.index
ones = np.ones(len(climate))
scalings = [np.maximum(rain-evap, 0)/X1, \
                        np.maximum(evap-rain, 0)/X1, ones]
scalings = np.column_stack(scalings)

# Numerical solver
LOGGER.info("Run model - radau")
niter, s1, sim = slow.numerical_model(fluxes, dfluxes,
                                      scalings, s0,
                                      timestep, method="Radau")
sim = param*np.abs(sim) * timestep

cns = [f"{fn}_radau" for fn in flux_names]
sim = pd.DataFrame(sim, columns=cns, index=rain.index)
sim.loc[:, "s_radau"] = s1

# Numerical solver, zoom on a particular time step
LOGGER.info("Run model - radau - zoom")
zdate = "2022-02-24"
idx = np.where(time_index == zdate)[0][0]
znsubsteps = 48
zscalings = np.repeat(scalings[[idx]], znsubsteps, axis=0)
zscalings[:, -1] = 1.
zs0 = s1[idx-1]
zniter, zs1, zsim = slow.numerical_model(fluxes, dfluxes,
                                      zscalings, zs0,
                                      timestep/znsubsteps, method="Radau")
zsim = param*np.abs(zsim) * timestep
ztimes = pd.date_range("2022-02-24", "2022-02-25", periods=znsubsteps+1)[1:]
zsim = pd.DataFrame(zsim, columns=cns, index=ztimes)
zsim.loc[:, "s_radau"] = zs1

# Get store at the beginning and end of timestep
LOGGER.info("Compute fluxes approx")
s_end = sim.s_radau
s_start = sim.s_radau.shift(1)
s_mid = (s_start+s_end)/2

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
    sim.loc[:, f"{cbase}_approx_midpt"] = X1*sgn*timestep*sc*fx(s_mid)

    # mid-point function
    sim.loc[:, f"{cbase}_approx_midptfun"] = X1*sgn*timestep*sc*(fx(s_end)+fx(s_start))/2

    # end-point
    #sim.loc[:, f"{cbase}_approx_endpt"] = X1*sgn*timestep*sc*fx(s_end)

LOGGER.info("Plot")
plt.close("all")

fig = plt.figure(figsize=(16, 15), layout="constrained")
mosaic = [[f"{fn}/ts", f"{fn}/scat" if fn != "s" else "."]
          for fn in ["s"] + flux_names]
kw = dict(width_ratios=[3, 1], hspace=0.08, wspace=0.08)
axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

for iax, (aname, ax) in enumerate(axs.items()):
    fxn, ptype = aname.split("/")
    df = sim.filter(regex=f"^{fxn}_", axis=1)
    df.columns = [re.sub(f"{fxn}_", "", cn)
                  for cn in df.columns]

    if ptype == "ts":
        xlabel = ""
        ylabel = "Store filling level [-]" if fxn == "s" \
            else "Flux [mm $day^{-1}$]"
        title = "simulated times series"

        for cn in df.columns:
            lw = 3 if cn == "radau" else 2
            se = df.loc["2022-02":"2022-03", cn]
            lab = approx_long_names[cn]
            ax.plot(se, lw=lw, color=colors[cn], label=lab)
    else:
        xlabel = "Radau flux computation [mm $day^{-1}$]" \
            if iax == len(axs) - 1 else ""
        ylabel = "Approximated flux [mm $day^{-1}$]"
        title = "approx. vs Radau"

        x = df.filter(regex="radau", axis=1).squeeze()
        for cn in df.columns:
            if cn == "radau":
                continue

            lab = approx_long_names[cn]
            ax.plot(x, df.loc[:, cn], "o", color=colors[cn], label=lab)

    title = f"({letters[iax]}) {flux_long_names[fxn]} - {title}"
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if ptype == "ts":
        loc = 7 if fxn == "s" else 1
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%y"))
    else:
        loc = 2
    ax.legend(loc=loc, framealpha=0.)

fp = fimg / f"approximate_flux_approximation.png"
fig.savefig(fp, dpi=fdpi)


# Plot zoom simulation
fig = plt.figure(figsize=(10, 8), layout="constrained")
mosaic = [[f"{fn}/ts"] for fn in ["s_radau", "eff_rain_radau"]]
kw = dict(hspace=0.08)
axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)
for iax, (aname, ax) in enumerate(axs.items()):
    varname, ptype = re.split("/", aname)
    ax.plot(zsim.loc[:, varname], "-o",
            color=colors["radau"], lw=3,
            label="Radau")

    # Decorate
    fxn = re.sub("_radau", "", varname)
    title = f"({letters[iax]}) {flux_long_names[fxn]}"
    ylabel = "Store filling level [-]" if fxn == "s"\
        else "Flux [mm]"
    ax.set(title=title, ylabel=ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y\n%H:%M"))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

    # Cumsum
    if varname == "eff_rain_radau":
        tax = ax.twinx()
        tax.plot(zsim.loc[:, varname].cumsum(), "-",
                 color=colors["radau"], lw=2,
                 linestyle="-.")
        ax.plot([], "-",
                 color=colors["radau"], lw=2,
                 linestyle="-.",
                 label="Radau - cumulative")

        ylabel = "Cumulative flux [mm]"
        tax.set(ylabel=ylabel)

    ax.legend(loc=7, framealpha=0.)

fp = fimg / f"approximate_flux_approximation_zoom.png"
fig.savefig(fp, dpi=fdpi)

LOGGER.completed()

