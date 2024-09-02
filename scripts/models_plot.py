#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2024-08-30 15:29:53.612017
## Comment : Plot model simulations
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
import tables

import matplotlib as mpl

# Select backend
mpl.use("Agg")

import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

import data_utils
import hdf5_utils

#----------------------------------------------------------------------
# @Config
#----------------------------------------------------------------------

parser = argparse.ArgumentParser(\
    description="Plot model simulations", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-e", "--extension", help="Image file extension", \
                    type=str, default="png")
args = parser.parse_args()

ode_methods = ["radau", "c_quasoare_3", "c_quasoare_5"]
iparam = 1

# Image file extension
imgext = args.extension

# Plot dimensions
fdpi = 120
awidth = 8
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
fdata = froot / "outputs" / "simulations"

fimg = froot / "images" / "simulations"
fimg.mkdir(exist_ok=True, parents=True)

#------------------------------------------------------------
# @Logging
#------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)
LOGGER.log_dict(vars(args), "Command line arguments")

#------------------------------------------------------------
# @Get data
#------------------------------------------------------------


#------------------------------------------------------------
# @Plot
#------------------------------------------------------------
lf = list(fdata.glob("*.hdf5"))
for f in lf:
    # Get model name and siteid
    taskid = int(re.sub(".*TASK", "", f.stem))
    model_name, siteid = data_utils.get_config(taskid)

    sims = {}
    with tables.open_file(f, "r") as h5:
        for ode_method in ode_methods:
            gr = h5.root[ode_method]
            m = re.sub("_", "", ode_method)
            tname = f"S{siteid}_{m}_param{iparam:03d}"
            sims[ode_method] = hdf5_utils.convert(gr[tname][:])

    plt.close("all")
    mosaic = [["s1"]*2]+[[f"flux{i+1}" for i in t] \
                        for t in np.array_split(np.arange(4), 2)]
    nrows, ncols = len(mosaic), len(mosaic[0])
    fig = plt.figure(figsize=(awidth*ncols, aheight*nrows), layout="tight")
    axs = fig.subplot_mosaic(mosaic)

    for iax, (aname, ax) in enumerate(axs.items()):
        for ode_method, sim in sims.items():
            se = sim.loc[:, aname].loc["2022"]
            lab = f"{ode_method}"
            se.plot(ax=ax, label=lab)

        ax.legend(loc=2, fontsize="x-small")

    ftitle = f"{siteid} - Model {model_name}"
    fig.suptitle(ftitle)

    # Save file
    fp = fimg / f"{f.stem}.{imgext}"
    fig.savefig(fp, dpi=fdpi, transparent=ftransparent)
    #putils.blackwhite(fp)

LOGGER.completed()
