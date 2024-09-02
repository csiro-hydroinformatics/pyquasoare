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
parser.add_argument("-m", "--model_name", help="Model", \
                    type=str, required=True)
parser.add_argument("-s", "--siteid", help="Siteid", \
                    type=str, required=True)
parser.add_argument("-i", "--iparam", help="Parameter number", \
                    type=int, required=True)
args = parser.parse_args()

ode_methods = ["radau", "c_quasoare_3", "c_quasoare_50"]
model_name_selected = args.model_name
siteid_selected = args.siteid
iparam_selected = args.iparam

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
    if model_name != model_name_selected or \
            siteid != siteid_selected:
        continue

    LOGGER.info(f"Plotting {model_name:4s}/{siteid} (TASK {taskid})")

    sims = {}
    info = {}
    with tables.open_file(f, "r") as h5:
        for ode_method in ode_methods:
            gr = h5.root[ode_method]
            m = re.sub("_", "", ode_method)
            tname = f"S{siteid}_{m}_param{iparam_selected:03d}"
            tb = gr[tname]
            sims[ode_method] = hdf5_utils.convert(tb[:])

            info[ode_method] = {
                            "s1_max": tb.attrs.s1_max, \
                            "alpha_max": tb.attrs.alpha_max
                        }
    info = pd.DataFrame(info)

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

