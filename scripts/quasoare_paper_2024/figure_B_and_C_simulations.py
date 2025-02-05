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
import matplotlib.ticker as ticker

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
fdpi = 300

awidth1 = 15
aheight1 = 6

awidth2 = 9
aheight2 = 7

# Figure transparency
ftransparent = False

col_nodes = "tab:red"

col_ana = "grey"
ls_ana = "--"
lw_ana = 5

col_qua = "tab:blue"
ls_qua = "-"
lw_qua = 3

col_rad = "tab:orange"
ls_rad = "-"
lw_rad = 6

col_err = "k"
lw_err = 1

# Set matplotlib options
#mpl.rcdefaults() # to reset
putils.set_mpl()

varnames = {
    "QR": {"flux2": "Outflow"}, \
    "CR": {"flux2": "Outflow"}, \
    "BCR": {"flux2": "Outflow"}, \
    "GR": {\
        "flux1": "Infiltrated rain", \
        "flux2": "Actual ET", \
        "flux3": "Percolation"
    }, \
    "GRM": {\
        "flux1": "Infiltrated rain", \
        "flux2": "Actual ET", \
        "flux3": "Percolation", \
        "flux4": "none"
    },
}

scaling_names = {
    "QR": {"flux2": "Qref"}, \
    "CR": {"flux2": "Qref"}, \
    "BCR": {"flux2": "Qref"}, \
    "GR": {\
        "flux1": "P", \
        "flux2": "E", \
        "flux3": r"$\theta$"
    }, \
    "GRM": {\
        "flux1": "P", \
        "flux2": "E", \
        "flux3": r"$\theta$",\
        "flux4": "(no scaling)"
    }
}

for m, vn in varnames.items():
    vn["s1"] = "Store level"

#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent
fdata = froot / "outputs" / "simulations"

fimg = froot / "images" / "figures"
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
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    # Format axis number as 2.3 x 10^5
    def fmt(x, pos):
        if abs(x) <= 1e-100:
            return "0"
        else:
            power = int(math.log(abs(x), 10))
            if power == 0:
                return f"{x:0.1f}"
            else:
                base = x / 10 ** power
                return f"${base:+0.1f}$\n  "\
                    + f"$\\times 10^{{{power}}}$"
    ax.yaxis.set_major_formatter(fmt)

lf = list(fdata.glob("*.hdf5"))

for f in lf:
    # Get model name and siteid
    taskid = int(re.sub(".*TASK", "", f.stem))
    model_name, siteid = data_utils.get_config(taskid)
    if model_name != model_name_selected or \
            siteid != siteid_selected:
        continue

    routing = not model_name.startswith("GR")

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
            theta = tb.attrs["param"]
            sims[ode_method] = hdf5_utils.convert(tb[:])

            cc = ["s1_min", "s1_max", "alpha1_min", "alpha1_max", \
                            "param"]
            info[ode_method] = {cn: tb.attrs[cn] if cn in tb.attrs \
                                            else np.nan for cn in cc}
            tname = f"S{siteid}_{m}_fluxes{iparam_selected:03d}"
            try:
                tb = gr[tname]
                alphas = tb.attrs["alphas"]
                mean_scalings = tb.attrs["mean_scalings"]
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
    axs2 = fig2.subplot_mosaic(mosaic[1:], sharex=True)

    for iax, (aname, ax1) in enumerate(axs1.items()):
        ax2 = None
        if aname in axs2:
            ax2 = axs2[aname]

        for ode_method, sim in sims.items():
            se = sim.loc[:, aname].loc[start:end]

            lab = re.sub("_", " ", re.sub("^(c|py)_", "", ode_method)).title()
            if re.search("Quaso", lab):
                lab = re.sub("Quasoare", "QuaSoARe", lab)

            if ode_method == "radau":
                ls, lw, col = ls_rad, lw_rad, col_rad
            else:
                ls, lw, col = ls_qua, lw_qua, col_qua

            se.plot(ax=ax1, label=lab, ls=ls, lw=lw, color=col)

            tax1, tax2 = None, None

            if ode_method == ode_method_selected:
                ser = sims["radau"].loc[se.index, aname]
                err = se-ser
                tax1 = ax1.twinx()
                err.plot(ax=tax1, lw=lw_err, color=col_err)
                format_errorax(tax1)
                ax1.plot([], lw=lw_err, color=col_err,
                         label="Error QuaSoARe-Radau")

            if ode_method == ode_method_selected and aname.startswith("flux")\
                                                and ode_method in fluxes:
                fx = fluxes[ode_method]
                s = fx["s"]
                fx_true = fx[f"{aname}_true"]
                fx_approx = fx[f"{aname}_approx"]

                ax2.plot(s, fx_true, "--", color=col_ana, lw=lw_ana, \
                                    label="True flux function")

                falphas = np.interp(alphas, s, fx_true)
                ax2.plot(alphas, falphas, "o", ms=12, color=col_nodes, \
                                    label="Interpolation nodes", zorder=100)

                m = " ".join(re.split("_", ode_method)[1:]).title()
                m = re.sub("Quasaore", "QuaSoARe", m)
                ax2.plot(s, fx_approx, lw=lw_qua, color=col_qua, label=f"{m} flux")

                unit = r"m$^3$ s$^{-1}$" if routing else r"mm day$^{-1}$"
                ax2.set_ylabel(f"Instantaneous flux [{unit}]")
                if iax==len(axs1)-1:
                    ax2.set_xlabel(r"Store filling level $S/\theta$ [-]")

                ifx = int(aname[-1])-1
                scaling = mean_scalings[ifx]
                sname = scaling_names[model_name][aname]
                sunit = "mm" if re.search("theta", sname) else unit
                if sname != "none":
                    txt = f"{sname} = {scaling:0.1f} {sunit}"
                else:
                    sname = "(no scaling)"

                ax2.text(0.02, 0.02, txt, transform=ax2.transAxes, \
                                            fontweight="bold")

                tax2 = ax2.twinx()
                tax2.plot(s, fx_approx-fx_true, lw=lw_err, color=col_err)
                format_errorax(tax2)
                ax2.plot([], lw=lw_err, color=col_err,
                         label="Interpolation error")

        if aname == "s1":
            ax1.legend(loc=3, fontsize="medium", framealpha=0.)

        if aname=="flux1":
            ax2.legend(loc=1, fontsize="small",
                       framealpha=0.)

        title = f"({letters[iax]}) {varnames[model_name][aname]}"
        ax1.set(title=title, xlabel="")

        unit = r"mm" if model_name.startswith("GR") \
                            else "m$^3$"
        unit = "-" if aname == "s1" else unit
        nm = "Store filling level $S/\\theta$" if aname=="s1" else "Total flux"
        ax1.set_ylabel(f"{nm} [{unit}]")
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(5))

        if not ax2 is None:
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(5))

        if not tax1 is None:
            tax1.set_ylabel(f"Error [{unit}]")

        if not tax2 is None:
            unit = "mm day$^{-1}$" if model_name.startswith("GR") else "m$^3$ s$^{-1}$"
            tax2.set_ylabel(f"Error [{unit}]")
            title = f"({letters[iax-1]}) {varnames[model_name][aname]}"
            ax2.set(title=title)


    # Save file
    theta = info.loc["param", "radau"]
    base = f"{f.stem}_{model_name}_{theta:0.0f}"

    meta = {\
            "alphas": alphas.tolist(), \
            "mean_scalings": mean_scalings.tolist()
            }
    fm = fimg / f"figure_C_{base}_meta.json"
    with fm.open("w") as fo:
        json.dump(meta, fo, indent=4)

    fp = fimg / f"figure_C_{base}_sim.{imgext}"
    fig1.savefig(fp, dpi=fdpi, transparent=ftransparent)
    fp = fimg / f"figure_B_{base}_fluxes.{imgext}"
    fig2.savefig(fp, dpi=fdpi, transparent=ftransparent)

    #putils.blackwhite(fp)

LOGGER.completed()

