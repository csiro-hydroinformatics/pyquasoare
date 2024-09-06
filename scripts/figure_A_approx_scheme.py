#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2024-04-20 18:25:07.580140
## Comment : Figure to show the approximate scheme
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

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

from pyrezeq import approx, models

#----------------------------------------------------------------------
# @Config
#----------------------------------------------------------------------

parser = argparse.ArgumentParser(\
    description="Figure to show the approximate scheme", \
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

nnalphas = [3, 4]

col_nodes = "tab:red"

col_ana = "grey"
ls_ana = "--"
lw_ana = 3

col_qua = "tab:blue"
ls_qua = "-"
lw_qua = 2

# Set matplotlib options
#mpl.rcdefaults() # to reset
#putils.set_mpl()

#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fimg = froot / "images" / "figures"
fimg.mkdir(exist_ok=True)

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
mosaic = [[f"dsdt_{n}", f"st_{n}"] for n in nnalphas]
fncols, fnrows = len(mosaic[0]), len(mosaic)
figsize = (awidth*fncols, aheight*fnrows)
fig = plt.figure(constrained_layout=True, figsize=figsize)

# Create mosaic with named axes
kw = dict(width_ratios=[1, 1.7], hspace=0.1, wspace=0.1)
axs = fig.subplot_mosaic(mosaic, sharey=True, gridspec_kw=kw)
iax = 0

for nalphas in nnalphas:
    # Piecewise approx
    amin, amax = 0., 1.
    alphas = np.linspace(amin, amax, nalphas)
    amat, bmat, cmat, _ = approx.quad_coefficient_matrix([flux_fun], alphas)

    u0 = 0.9
    tmax = 10.
    nval = 100
    times = np.linspace(0, tmax, nval)
    timestep = times[1]-times[0]
    scalings = np.ones((nval, 1))
    niter, s, sim = models.quad_model(alphas, scalings, amat, bmat, cmat,\
                                        u0, timestep)
    s = np.insert(s, 0, u0)[: -1]
    res = pd.DataFrame({"time": times, "quasoare": s})
    res.loc[:, "quasoare_diff"] = approx.quad_fun_from_matrix(alphas, amat,
                                                              bmat, cmat, s)

    res.loc[:, "analytical"] = u0/np.sqrt(1+u0**2*times)
    res.loc[:, "analytical_diff"] = -u0**3/(1+u0**2*times)**1.5/2.


    LOGGER.info("Drawing plot")
    anames = [f"dsdt_{nalphas}", f"st_{nalphas}"]
    for aname in anames:
        ax = axs[aname]
        if aname.startswith("dsdt"):
            title = f"Piecewise interpolation - {nalphas} nodes"
            xlab = r"$\frac{dS}{dt}$" if nalphas==nnalphas[-1] else ""
            ylab = r"$S(t)$"

            x = res.analytical_diff
            y = res.analytical

            x0 = flux_fun(u0)
            y0 = u0

            yy = np.linspace(amin, amax, 500)
            xx = flux_fun(yy)

            lab = "True reservoir equation"
            ax.plot(xx, yy, color=col_ana, label=lab, \
                                    linestyle=ls_ana, lw=lw_ana)

            lab = "Interpolation nodes"
            ax.plot([flux_fun(a) for a in alphas], alphas, "s",\
                            color=col_nodes, label=lab, zorder=100)

            axi = ax.inset_axes([0.62, 0.78, 0.38, 0.22])
            axi.plot(res.quasoare_diff-x, y, "-", color=col_qua)
            axi.text(0.97, 0.03, "Interpolation error", fontsize=7, \
                    transform=axi.transAxes, va="bottom", ha="right")

            xlim = axi.get_xlim()
            xm = np.abs(xlim).max()*1.1
            axi.set(ylim=(0., 1.), xlim=(-xm, xm), ylabel=ylab, yticks=[])
            for tk in axi.xaxis.get_major_ticks():
                tk.label1.set_fontsize(7)
            putils.line(axi, 0, 1, 0, 0, "k-", lw=0.5)


        else:
            title = f"Solution of reservoir equation - {nalphas} nodes"
            xlab = r"$t$" if nalphas==nnalphas[-1] else ""
            ylab = ""

            x = res.time
            y = res.quasoare

            x0 = 0
            y0 = u0

            lab = "Analytical solution"
            ax.plot(x, res.analytical, label=lab, lw=lw_ana, linestyle=ls_ana, \
                    color="0.5", zorder=100)


            axi = ax.inset_axes([0.76, 0.78, 0.24, 0.22])
            axi.plot(x, y-res.analytical, "-", color=col_qua)
            axi.text(0.03, 0.97, "Simulation error", fontsize=7, \
                    transform=axi.transAxes, va="top", ha="left")

            ylim = axi.get_ylim()
            ym = np.abs(ylim).max()*1.3
            axi.set(ylim=(-ym, ym), xlabel=r"$t$", xticks=[])
            for tk in axi.yaxis.get_major_ticks():
                tk.label1.set_fontsize(7)
            putils.line(axi, 1, 0, 0, 0, "k-", lw=0.5)


        ax.plot(x0, y0, "o", color=col_qua, label="Initial condition")
        ax.plot(x, y, linestyle=ls_qua, color=col_qua, lw=lw_qua, \
                                label="QUASOARE solution")

        title = f"({letters[iax]}) {title}"
        ax.set(title=title)
        ax.set_xlabel(xlab, fontsize=18)
        ax.set_ylabel(ylab, fontsize=18)

        if nalphas==nnalphas[0]:
            leg = ax.legend(loc=3, fontsize="small", \
                        framealpha=1.0)
            if len(nnalphas)>1:
                ax.set(xticks=[])

        for ia, a in enumerate(alphas):
            putils.line(ax, 1, 0, 0, a, ":", color="0.5", lw=0.9)
            if nalphas==nnalphas[0] and ia==0 and aname.startswith("dsdt"):
                continue
            props = dict(boxstyle="round", fc="w", ec="none", alpha=0.9)
            xtxt = -0.35 if aname.startswith("dsdt") else 5.
            ax.text(xtxt, a, f"$\\alpha_{ia+1} = {a:0.2f}$".format(ia=ia, a=a), \
                        va="center", ha="center", color="0.5", bbox=props, \
                        zorder=200)

        iax += 1

# Save file
fp = fimg / f"figure_A_approx_scheme.{imgext}"
fig.savefig(fp, dpi=fdpi, transparent=ftransparent)
putils.blackwhite(fp)



LOGGER.completed()
