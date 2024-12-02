#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2024-08-30 15:29:53.612017
## Comment : Plot performance
##
## ------------------------------


import sys, os, re, json, math
import argparse
from pathlib import Path

#import warnings
#warnings.filterwarnings("ignore")

from datetime import datetime

from dateutil.relativedelta import relativedelta as delta
from string import ascii_lowercase as letters

import numpy as np
import pandas as pd

import matplotlib as mpl

# Select backend
mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

import data_utils

#----------------------------------------------------------------------
# @Config
#----------------------------------------------------------------------

parser = argparse.ArgumentParser(\
    description="Plot performance", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-e", "--extension", help="Image file extension", \
                    type=str, default="png")
args = parser.parse_args()

siteids = data_utils.SITEIDS
model_names = ["CR", "BCR", "GR", "GRM"]
ode_methods = data_utils.ODE_METHODS

x_metric = "ERRABSMAX_INTERP_FLUX"
y_metric = "ERRABSMAX_SIM_FLUX"

# Image file extension
imgext = args.extension

# Plot dimensions
fdpi = 100 #300
awidth = 6
aheight = 4

# Figure transparency
ftransparent = False

# Set matplotlib options
#mpl.rcdefaults() # to reset
putils.set_mpl()

#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()

froot = source_file.parent.parent
fdata = froot / "outputs"

fimg = froot / "images" / "figures"
fimg.mkdir(exist_ok=True, parents=True)

#------------------------------------------------------------
# @Logging
#------------------------------------------------------------
basename = source_file.stem
flogs = froot / "logs"
flogs.mkdir(exist_ok=True)
flog = flogs / f"{source_file.stem}.log"
LOGGER = iutils.get_logger(basename, flog=flog)
LOGGER.log_dict(vars(args), "Command line arguments")

#------------------------------------------------------------
# @Get data
#------------------------------------------------------------
fr = fdata / "results.csv"
results, _ = csv.read_csv(fr, dtype={"siteid": str})
idx = results.ode_method.str.contains("c_.*quasoare_10", regex=True)
results = results.loc[idx]

#------------------------------------------------------------
# @Plot
#------------------------------------------------------------

plt.close("all")

mosaic = [["."]+[f"title/{fx+1}" for fx in range(4)]]
mosaic += [[f"{mo}/title"]+[f"{mo}/{fx+1}" for fx in range(4)]\
                for mo in model_names]

fnrows = len(mosaic)
fncols = len(mosaic[0])

# Create figure
figsize = (awidth*fncols, aheight*fnrows)
fig = plt.figure(constrained_layout=True, figsize=figsize)

gw = dict(wspace=0.1, hspace=0.1, width_ratios=[0.2]+[5]*(fncols-1), \
                        height_ratios=[0.2]+[4]*(fnrows-1))
axs = fig.subplot_mosaic(mosaic, gridspec_kw=gw)
mleft = re.sub(".*/", "", mosaic[0][0])
iplot = 0

for iax, (aname, ax) in enumerate(axs.items()):
    model_name, fx = re.split("/", aname)

    if fx=="title":
        ax.text(0.5, 0.5, model_name, rotation=90, \
                    fontweight="bold", ha="left", va="center", \
                    fontsize=25)
        ax.axis("off")
        continue

    if model_name=="title":
        ax.text(0.5, 0.5, f"Flux {fx}", \
                    fontweight="bold", ha="center", \
                    va="center", fontsize=25)
        ax.axis("off")
        continue


    idx = results.model_name==model_name
    if idx.sum()==0:
        ax.axis("off")
        continue

    x = results.loc[idx, f"{x_metric}{fx}"]
    y = results.loc[idx, f"{y_metric}{fx}"]
    ax.plot(x, y, "o")


    title = f"({letters[iplot]})"
    ax.text(0.02, 0.98, title, fontweight="bold", fontsize=18, \
                    va="top", ha="left", transform=ax.transAxes)

    iplot += 1

# Save file
fp = fimg / f"figure_E_error_analysis.{imgext}"
fig.savefig(fp, dpi=fdpi, transparent=ftransparent)
#putils.blackwhite(fp)

LOGGER.completed()
