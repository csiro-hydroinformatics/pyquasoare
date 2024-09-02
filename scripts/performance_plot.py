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

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils, violinplot

import data_utils

import importlib
importlib.reload(violinplot)

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
model_names = data_utils.MODEL_NAMES
ode_methods = data_utils.ODE_METHODS

metrics = {
    "ERRABSMAX_FLUX": "Max absolute error", \
    "ERRBAL_FLUX[%]": "Mass balance error", \
    "RUNTIME_RATIO[%]": "Runtime ratio", \
    "NITER_RATIO[%]": "Iteration ratio"
}

# Image file extension
imgext = args.extension

# Plot dimensions
fdpi = 120
awidth = 7
aheight = 5

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

fimg = froot / "images"
fimg.mkdir(exist_ok=True)

#------------------------------------------------------------
# @Logging
#------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)
LOGGER.log_dict(vars(args), "Command line arguments")

#------------------------------------------------------------
# @Get data
#------------------------------------------------------------
fr = fdata / "results.csv"
results, _ = csv.read_csv(fr, dtype={"siteid": str})
idx = ~results.ode_method.str.contains("radau|^c_", regex=True)
idx = results.ode_method!="analytical"
results = results.loc[idx]

results.loc[:, "ERRABSMAX_FLUX"] = results.filter(regex="ERRABSMAX", \
                                            axis=1).max(axis=1)
results.loc[:, "ERRBAL_FLUX[%]"] = results.filter(regex="ERRBAL", \
                                            axis=1).max(axis=1)

#------------------------------------------------------------
# @Plot
#------------------------------------------------------------

plt.close("all")

mosaic = [[f"{mo}/{me}" for me in metrics] for mo in model_names]
fnrows = len(mosaic)
fncols = len(mosaic[0])

# Create figure
figsize = (awidth*fncols, aheight*fnrows)
fig = plt.figure(constrained_layout=True, figsize=figsize)

gw = dict(height_ratios=[1]*fnrows, width_ratios=[1]*fncols)
axs = fig.subplot_mosaic(mosaic, gridspec_kw=gw)
mleft = re.sub(".*/", "", mosaic[0][0])

for iax, (aname, ax) in enumerate(axs.items()):
    model_name, metric = re.split("/", aname)
    idx = results.model_name==model_name
    if idx.sum()==0:
        ax.axis("off")
        continue

    df = pd.pivot_table(results.loc[idx], \
                    index=["siteid", "iparam"], columns="ode_method", \
                    values=metric)
    df.columns = [re.sub("_", "\n", re.sub("py_", "", cn)) \
                            for cn in df.columns]
    df = np.log10(1e-10+df)

    #if not "analytical" in df.columns:
    #    df.loc[:, "analytical"] = np.nan
    cc = ["rk45", "quasoare\n3", "quasoare\n5", "quasoare\n50"]
    df = df.loc[:, cc]

    worst = df.loc[:, "quasoare\n3"].idxmax()
    LOGGER.info(f"{model_name}/{metric} quasoare_3 worst: {worst}")

    vl = violinplot.Violin(df, show_text=False)
    vl.draw(ax=ax)

    meds = vl.stats.loc["median", :]
    for i, m in enumerate(meds):
        ax.text(i, m, f"{10**m:0.1e}", \
                    fontweight="bold", \
                    va="bottom", ha="center")

    title = f"({letters[iax]}) {metrics[metric]}"
    ax.set_title(title)

    ytk = np.unique(np.round(ax.get_yticks(), 0))
    ytxt = [re.sub("-0", "0", r"$10^{{{v:0.0f}}}$".format(v=v)) for v in ytk]
    ax.set_yticks(ytk)
    ax.set_yticklabels(ytxt)

    ylab = ""
    if metric == mleft:
        ylab= f"{model_name}\n"

    ylab += re.sub("_", " ", metric.title())
    ax.set_ylabel(ylab)

# Save file
fp = fimg / f"performance.{imgext}"
fig.savefig(fp, dpi=fdpi, transparent=ftransparent)
#putils.blackwhite(fp)

LOGGER.completed()
