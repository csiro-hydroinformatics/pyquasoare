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

metrics = {
    "ERRABSMAX_SIM_FLUX": "Max absolute simulation error $E^m$", \
    "ERRBAL_SIM_FLUX[%]": "Mass balance simulation error $B^m$", \
    "RUNTIME_RATIO[%]": "Runtime ratio $R^m$", \
    #"NITER_RATIO[%]": "Iteration ratio"
}

# Select ode method to include in plot
ode_method_selected = ["RK45", "QuaSoARe\n10", "QuaSoARe\n50", "QuaSoARe\n500"]
ode_method_worst = "QuaSoARe\n10"


# Image file extension
imgext = args.extension

# Plot dimensions
fdpi = 200 #300
awidth = 15 if ode_method_selected is None else 6
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

froot = source_file.parent.parent.parent
fdata = froot / "outputs"

fimg = froot / "images" / "figures"
fimg.mkdir(exist_ok=True, parents=True)

#------------------------------------------------------------
# @Logging
#------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

#------------------------------------------------------------
# @Get data
#------------------------------------------------------------
fr = fdata / "results.csv"
results, _ = csv.read_csv(fr, dtype={"siteid": str})
idx = ~results.ode_method.str.startswith("c_")
idx &= results.ode_method != "analytical"
results = results.loc[idx]

results.loc[:, "ERRABSMAX_SIM_FLUX"] = results.filter(regex="ERRABSMAX_SIM", \
                                            axis=1).max(axis=1)
rbal = results.filter(regex="ERRBAL", axis=1)
results.loc[:, "ERRBAL_SIM_FLUX[%]"] = rbal.max(axis=1)

#------------------------------------------------------------
# @Plot
#------------------------------------------------------------

plt.close("all")

mosaic = [["."]+[f"title/{me}" for me in metrics]]
mosaic += [[f"{mo}/title"]+[f"{mo}/{me}" for me in metrics] for mo in model_names]

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
    model_name, metric = re.split("/", aname)

    if metric=="title":
        ax.text(0.5, 0.5, f"{model_name} model", rotation=90, \
                    fontweight="bold", ha="left", va="center", \
                    fontsize=25)
        ax.axis("off")
        continue

    if model_name=="title":
        ax.text(0.5, 0.5, metrics[metric], \
                    fontweight="bold", ha="center", \
                    va="center", fontsize=25)
        ax.axis("off")
        continue


    idx = results.model_name==model_name
    idx &= results.loc[:, metric].notnull()
    if idx.sum()==0:
        ax.axis("off")
        continue

    ylog = bool(re.search("FLUX", metric))

    df = pd.pivot_table(results.loc[idx], \
                    index=["siteid", "iparam"], columns="ode_method", \
                    values=metric)
    df.columns = [re.sub("_", "\n", re.sub("py_", "", cn)).upper() \
                            for cn in df.columns]
    df.columns = [re.sub("QUASOARE", "QuaSoARe", cn) for cn in df.columns]

    if ylog:
        df = np.log10(1e-100+df)

    if not ode_method_selected is None:
        selected = ode_method_selected
        if model_name == "QR":
            selected = ["RADAU"]+selected
    else:
        selected = df.columns

    df = df.loc[:, selected]

    worst = df.loc[:, ode_method_worst].idxmax()
    m = re.sub("\n", " ", ode_method_worst)
    LOGGER.info(f"{model_name}/{metric} {m} worst: {worst}")

    v0 = df.min()
    v1 = df.median()
    v2 = df.max()
    x = np.arange(len(v0))
    width = 0.7
    ax.bar(x, v2-v0, bottom=v0, \
                color="tab:blue", alpha=1.0, width=width)

    ax.set_xticks(x)
    ax.set_xticklabels(v0.index)

    for xx, vv in zip(x, v1.values):
        v10 = 10**vv if ylog else vv
        fmt = "0.1e" if ylog else "0.1f"
        ax.plot([xx-width/2, xx+width/2], [vv]*2, "k-", lw=2)
        ax.text(xx, vv, "{v10:{fmt}}".format(v10=v10,fmt=fmt),
                va="bottom", fontsize=21,
                ha="center", color="k", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=4,
                                            foreground="w")])

    title = f"({letters[iplot]})"
    ax.text(0.02, 0.98, title, fontweight="bold", fontsize=18, \
                    va="top", ha="left", transform=ax.transAxes)

    if ylog:
        ytk = np.unique(np.round(ax.get_yticks(), 0))
        ytxt = [re.sub("-0", "0", r"$10^{{{v:0.0f}}}$".format(v=v)) for v in ytk]
        ax.set_yticks(ytk)
        ax.set_yticklabels(ytxt)

    if re.search("^ERRABS", metric):
        ylab = r"Error [mm day$^{-1}$]" if model_name.startswith("GR") \
                        else r"Error [m3 s$^{-1}$]"
    elif re.search("^ERRBAL", metric):
        ylab = "Error [%]"
    else:
        ylab = "Ratio [%]"
    ax.set_ylabel(ylab)
    iplot += 1

# Save file
fp = fimg / f"figure_D_performance.{imgext}"
fig.savefig(fp, dpi=fdpi, transparent=ftransparent)
#putils.blackwhite(fp)

LOGGER.completed()
