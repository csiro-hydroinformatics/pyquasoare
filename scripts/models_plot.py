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
parser.add_argument("-o", "--ode_method", help="ODE integration method", \
                    type=str, default="c_quasoare_10")
parser.add_argument("-i", "--iparam", help="Parameter number", \
                    type=int, required=True)
args = parser.parse_args()

ode_method_selected = args.ode_method
ode_methods = ["radau", ode_method_selected]

model_name_selected = args.model_name
siteid_selected = args.siteid
iparam_selected = args.iparam

start = "2022"
end = "2022"

# Image file extension
imgext = args.extension

# Plot dimensions
fdpi = 120

awidth1 = 15
aheight1 = 6

awidth2 = 8
aheight2 = 6

# Figure transparency
ftransparent = False

# Set matplotlib options
#mpl.rcdefaults() # to reset
putils.set_mpl()

varnames = {
    "QR": {"flux1": "Outflow"}, \
    "CR": {"flux1": "Ooutflow"}, \
    "BCR": {"flux1": "Outflow"}, \
    "GRP": {\
        "flux1": "Infiltrated rain", \
        "flux2": "Actual ET", \
        "flux3": "Percolation"
    }, \
    "GRPM": {\
        "flux1": "Infiltrated rain", \
        "flux2": "Actual ET", \
        "flux3": "Percolation", \
        "flux4": "Recharge"
    },
}

for m, vn in varnames.items():
    vn["s1"] = "Store level"

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
def format_errorax(ax):
    ylim = ax.get_ylim()
    ym = np.abs(ylim).max()*1.5
    ax.set_ylim((-ym, ym))
    putils.line(ax, 1, 0, 0, 0, color="grey", linestyle=":", lw=0.8)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

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
    fluxes = {}
    info = {}
    with tables.open_file(f, "r") as h5:
        for ode_method in ode_methods:
            gr = h5.root[ode_method]
            m = re.sub("_", "", ode_method)
            tname = f"S{siteid}_{m}_sim{iparam_selected:03d}"
            tb = gr[tname]
            sims[ode_method] = hdf5_utils.convert(tb[:])

            cc = ["s1_min", "s1_max", "alpha1_min", "alpha1_max", \
                            "param"]
            info[ode_method] = {cn: tb.attrs[cn] if cn in tb.attrs \
                                            else np.nan for cn in cc}
            tname = f"S{siteid}_{m}_fluxes{iparam_selected:03d}"
            try:
                tb = gr[tname]
                fluxes[ode_method] = hdf5_utils.convert(tb[:])
            except:
                pass

    info = pd.DataFrame(info)


    plt.close("all")
    vns = np.sort([vn for vn in varnames[model_name]\
                        if vn != "s1"])
    mosaic = [["s1"]]+[[vn] for vn in vns]

    nrows, ncols = len(mosaic), len(mosaic[0])
    fig1 = plt.figure(figsize=(awidth1*ncols, aheight1*nrows), layout="tight")
    axs1 = fig1.subplot_mosaic(mosaic)

    fig2 = plt.figure(figsize=(awidth2*ncols, aheight2*(nrows-1)), \
                                        layout="tight")
    axs2 = fig2.subplot_mosaic(mosaic[1:])

    for iax, (aname, ax1) in enumerate(axs1.items()):
        if aname in axs2:
            ax2 = axs2[aname]

        for ode_method, sim in sims.items():
            se = sim.loc[:, aname].loc[start:end]

            lab = re.sub("_", " ", re.sub("^(c|py)_", "", ode_method))
            lw = 5 if ode_method == "radau" else 3
            se.plot(ax=ax1, label=lab, lw=lw)

            tax1, tax2 = None, None

            if ode_method == ode_method_selected:
                ser = sims["radau"].loc[se.index, aname]
                err = se-ser
                tax1 = ax1.twinx()
                err.plot(ax=tax1, lw=1.0, color="grey")
                format_errorax(tax1)

            if ode_method == ode_method_selected and aname.startswith("flux")\
                                                and ode_method in fluxes:
                fx = fluxes[ode_method]
                s = fx["s"]
                fx_true = fx[f"{aname}_true"]
                fx_approx = fx[f"{aname}_approx"]
                ax2.plot(s, fx_true, label="True flux")
                m = " ".join(re.split("_", ode_method)[1:])
                ax2.plot(s, fx_approx, label=f"{m} flux")
                ax2.legend(loc=2, fontsize="large")

                unit = "day$^{-1}$" if model_name.startswith("GR") \
                                else "sec^{-1}"
                ax2.set_ylabel(f"Instantanous flux [{unit}]")
                ax2.set_xlabel(r"Normalised store level $u$ [-]")

                tax2 = ax2.twinx()
                tax2.plot(s, fx_approx-fx_true, lw=1.0, color="grey")
                format_errorax(tax2)

        if aname == "s1":
            ax1.legend(loc=2, fontsize="large")


        title = varnames[model_name][aname]
        ax1.set(title=title, xlabel="")

        unit = "mm/day" if model_name.startswith("GR") else "m3/sec"
        nm = "Store level" if aname=="s1" else "Flux"
        ax1.set_ylabel(f"{nm} [{unit}]")

        if not tax1 is None:
            tax1.set_ylabel(f"Error [{unit}]")
        if not tax2 is None:
            tax2.set_ylabel(f"Error [{unit}]")


    # Save file
    theta = info.loc["param", "radau"]
    base = f"{f.stem}_{model_name}_{theta:0.0f}"
    fp = fimg / f"{base}_sim.{imgext}"
    fig1.savefig(fp, dpi=fdpi, transparent=ftransparent)
    fp = fimg / f"{base}_fluxes.{imgext}"
    fig2.savefig(fp, dpi=fdpi, transparent=ftransparent)

    #putils.blackwhite(fp)

LOGGER.completed()

