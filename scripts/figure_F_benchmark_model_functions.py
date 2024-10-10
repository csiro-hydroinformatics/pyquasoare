#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2024-10-09 Wed 03:16 PM
## Comment : Figure showing the 4 benchmark function models
##
## ------------------------------


import sys, os, re, json, math
import argparse
from pathlib import Path

from datetime import datetime

from string import ascii_lowercase as letters

import numpy as np
import pandas as pd

import matplotlib as mpl

# Select backend
mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TABLEAU_COLORS as colors

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

from pyquasoare import benchmarks

#----------------------------------------------------------------------
# @Config
#----------------------------------------------------------------------

parser = argparse.ArgumentParser(\
    description="Figure to show the benchmark model functions", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-e", "--extension", help="Image file extension", \
                    type=str, default="png")
args = parser.parse_args()
imgext = args.extension

# Plot dimensions
fdpi = 300
awidth = 5.5
aheight = 3.5

# Figure transparency
ftransparent = False

linestyles = ["-", "--", "-.", ":"]
colors = [c for n, c in colors.items()]

# Set matplotlib options
#mpl.rcdefaults() # to reset
#putils.set_mpl()

#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fimg = froot / "images" / "figures"
fimg.mkdir(exist_ok=True, parents=True)

#------------------------------------------------------------
# @Logging
#------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)
LOGGER.log_dict(vars(args), "Command line arguments")

#------------------------------------------------------------
# @Plot
#------------------------------------------------------------

# Reservoir function
flux_fun = lambda x: -x**3/2

# Create figure
plt.close("all")
mosaic = [["CR", "BCR"], ["GR", "GRM"]]
fncols, fnrows = len(mosaic[0]), len(mosaic)
figsize = (awidth*fncols, aheight*fnrows)
fig = plt.figure(constrained_layout=True, figsize=figsize)

# Create mosaic with named axes
kw = dict(width_ratios=[1, 1], hspace=0.1, wspace=0.1)
axs = fig.subplot_mosaic(mosaic, sharey=True, gridspec_kw=kw)


for iax, (model_name, ax) in enumerate(axs.items()):
    routing = False
    if model_name in ["CR", "BCR"]:
        routing = True
        nu = 3. if model_name == "CR" else 6.
        fluxes, _ = benchmarks.nonlinrouting_fluxes_noscaling(nu)
        names = ["Inflow", "Outflow"]

        # using data from 203024
        scalings = []

        unit = r"m$^3$ s$^{-1}$"

    elif model_name == "GR":
        fluxes, _ = benchmarks.gr4jprod_fluxes_noscaling()
        names = ["Infilt. Rain", "Actual ET", "Percolation"]

        # using data from 203024
        scalings = []

        unit = r"mm day$^{-1}$"

    elif model_name == "GRM":
        fpr = lambda x: (1.-x**3*(10-15*x+6*x**2)) if x>0 else 1.
        fae = lambda x: -(16*(x-0.5)**5+0.5) if x<1. else 4.-5.*x
        fperc = lambda x: -0.1*x**7 if x>0 else 0.
        fgw = lambda x: -0.05*x/(1+10*x) if x>0 else 0.
        fluxes = [fpr, fae, fperc, fgw]
        names = ["Infilt. Rain", "Actual ET", "Percolation", "Recharge"]

        # Using data from
        scalings = []

        unit = r"mm day$^{-1}$"

    x = np.linspace(0, 1, 100)
    tax = ax.twinx()
    for ifx, f in enumerate(fluxes):
        y = [f(xx)*scalings[ifx] for xx in x]
        lab = f"$f_{ifx+1}$ ({names[ifx]})"

        axx, addleg = (tax, True) if not routing and ifx>1 else (ax, False)
        axx.plot(x, y, linestyle=linestyles[ifx], lw=2, label=lab,\
                                    color=colors[ifx])
        if addleg:
            ax.plot([], [], linestyle=linestyles[ifx], lw=2, label=lab, \
                                    color=colors[ifx])
            axx.yaxis.set_major_locator(ticker.MaxNLocator(5))

    loc = 3 if routing else 10
    ax.legend(loc=loc)
    title = f"({letters[iax]}) Model {model_name}"
    ax.set(xlabel=r"Storage level $S/\theta$", ylabel=f"Fluxes [{unit}]", title=title)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.grid(axis="y")

# Save file
fp = fimg / f"figure_F_benchmark_model_functions.{imgext}"
fig.savefig(fp, dpi=fdpi, transparent=ftransparent)
putils.blackwhite(fp)



LOGGER.completed()
