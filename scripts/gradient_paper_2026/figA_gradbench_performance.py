#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-08-18 15:33:46.478450
## Comment : Figure presenting gradient benchmarking
##
## ------------------------------

import sys
import re
import json
import math
from itertools import product as prod
import argparse
from datetime import datetime
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
from string import ascii_lowercase as letters

import matplotlib as mpl

import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils, violinplot

# Package to plot spatial data
from hydrodiy.io import csv, iutils

# Select backend
mpl.use("Agg")


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "outputs" / f"gradbench_v{config.version}"
    fimg = froot / "images" / "manuscript" / "gradient_benchmarking"
    ScriptPaths = namedtuple("ScriptPaths",
                             ["source_file", "basename",
                              "froot", "fdata", "fimg"])
    script_paths = ScriptPaths(source_file, source_file.stem,
                               froot, fdata, fimg)
    fimg.mkdir(exist_ok=True, parents=True)

    for f in fimg.glob("*.png"):
        f.unlink()

    return script_paths


def get_logger(config, script_paths):
    basename = script_paths.basename
    logger = iutils.get_logger(basename, console=True)
    logger.log_dict(config._asdict(), "Configuration")
    logger.info("", nret=1)
    return logger


def get_data(config, script_paths, logger):
    fp = script_paths.fdata / f"performance_v{config.version}.csv"
    perf, _ = csv.read_csv(fp)
    data = namedtuple("Data", ["perf"])(perf)
    return data


def process(config, script_paths, logger, data):

    # Get data
    perf = data.perf

    gms = perf.TASK_OPT_gamma_mean.unique()
    epss = perf.TASK_OPT_findiff_log10eps.unique()
    nas = perf.TASK_OPT_nalphas.unique()

    gm_ref = "500_50_500"
    eps_ref = -5
    na_ref = 50

    for na, gm, eps in prod(nas, gms, epss):
        match = int(eps == eps_ref) + int(gm == gm_ref) + int(na == na_ref)
        if match < 2:
            continue

        logger.info(f"Plotting na={na} gm={gm} eps={eps}", nret=1)

        # Create figure
        plt.close("all")
        nrs = [1, 5, 50]
        metrics = ["runtimedelta", "errmax00", "errmax90"]
        mosaic = [["."] + [f"{nr}/title" for nr in nrs]] \
            + [[f"title/{vn}"] + [f"{nr}/{vn}" for nr in nrs]
               for vn in metrics]

        fncols, fnrows = len(mosaic[0]), len(mosaic)
        awidth, aheight = config.awidth, config.aheight
        figsize = (awidth*fncols, aheight*fnrows)
        fig = plt.figure(constrained_layout=True,
                         figsize=figsize)

        gw = dict(height_ratios=[1] + [6]*(fnrows - 1),
                  width_ratios=[1] + [6] *(fncols - 1))
        axs = fig.subplot_mosaic(mosaic, gridspec_kw=gw)

        nstations = len(perf.TASK_OPT_stationid.unique())
        nrepeat = len(perf.REPEAT.unique())
        iplot = 0

        idx0 = perf.TASK_OPT_findiff_log10eps == eps
        idx0 &= perf.TASK_OPT_nalphas == na
        idx0 &= perf.TASK_OPT_gamma_mean == gm

        for iax, (aname, ax) in enumerate(axs.items()):
            if re.search("title", aname):
                ax.axis("off")
                t1, t2 = aname.split("/")
                if t2 == "title":
                    title = f"{t1} reach" if t1 == "1" else f"{t1} reaches"

                    idx = idx0 & (perf.TASK_OPT_nreaches == int(t1))
                    nr = (perf.loc[idx, "nreservoirs"].unique()).squeeze()
                    nv = (perf.loc[idx, "nvariables"].unique()).squeeze()
                    ny = (perf.loc[idx, "ntimesteps"].unique() / 365.25).squeeze()
                    subtitle = f"{nr:0.0f} Reservoirs  {nv:0.0f} Variables" \
                        + f"  {ny:0.0f} Years"
                    ax.text(0.5, 0.1, subtitle,
                            va="center", ha="center",
                            fontsize="large")
                    rot = 0.
                else:
                    if re.search("delta", t2):
                        title = "Runtime ratio [-]"
                    elif re.search("max90", t2):
                        title = "Relative error\nabove 90th prc [-]"
                    else:
                        title = "Relative error [-]"
                    rot = 90.

                ax.text(0.5, 0.5, title,
                        va="center", ha="center",
                        fontweight="bold",
                        fontsize="x-large",
                        rotation=rot)

                continue

            nreaches, metric = aname.split("/")
            nreaches = int(nreaches)
            idx = idx0 & (perf.TASK_OPT_nreaches == nreaches)
            logger.info(f"Plot reaches={nreaches}/{metric}", ntab=1)

            pat = f"{metric}"
            cc = [cn for cn in perf.columns if re.search(pat, cn)]
            is_log = bool(re.search("log10", cc[0]))
            df = perf.loc[idx, cc]

            if re.search("runtimedelta", metric):
                quas_dt = perf.loc[idx, "log10runtime_neutral[sec]"].mean()
                txt = f"QuaSoARe sim runtime:\n"\
                    + f"{10**quas_dt:0.1e} sec/yr/reservoir"
                ax.text(0.98, 0.02, txt, va="bottom", ha="right",
                        transform=ax.transAxes, fontstyle="italic")

            logger.info(f"len(df) = {len(df)}", ntab=1)

            if len(df) != nrepeat * nstations:
                logger.info("no data. continue", ntab=1)
                ax.axis("off")
                continue

            cc = []
            for cn in df.columns:
                if re.search("_ref", cn):
                    cc.append("QuaSoARe\nGradient")
                else:
                    o = re.search("(?<=diff)\\d", cn).group()
                    cc.append(f"Fin.diff.\norder {o}")
            df.columns = cc

            try:
                vl = violinplot.Violin(df)
            except:
                continue
            vl.draw(ax=ax)

            if is_log:
                # Relabel violins to show log values
                for nm, elem in vl.elements.items():
                    txt = elem["median-text"]
                    y = 10**txt._y
                    txt.set_text(f"{y:0.1e}")

            # Axis
            title = f"({letters[iplot]})"
            ax.set_title(title, x=0.02, y=0.95,
                         va="top", ha="left",
                         fontsize="x-large")

            ax.set(xlabel="",
                   ylabel="")

            if is_log:
                if re.search("runtimedelta", metric):
                    ax.set_ylim((-0.5, 3.5))
                    tk = np.arange(0, 4)
                else:
                    ax.set_ylim((-6.5, 0.5))
                    tk = np.arange(-6, 2, 2)

                labs = ["1" if k == 0 else f"$10^{{{k}}}$" for k in tk]
                ax.set_yticks(tk, labels=labs)

            ax.grid(axis="y")

            iplot += 1

        # Save file
        cfg = f"N{na}_E{eps}_G{gm}"
        if match == 3:
            cfg += "_REF"
        fp = script_paths.fimg / \
                f"figA_gradbench_v{config.version}_{cfg}.{config.imgext}"
        fig.savefig(fp, dpi=config.fdpi, transparent=config.ftransparent)
        #putils.blackwhite(fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table presenting gradient benchmarking",
                                     formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version",
                        help="Version number",
                        type=str, required=True)
    args = parser.parse_args()

    # Config
    version = args.version
    awidth = 3
    aheight = 3
    fdpi = 300
    imgext = "png"
    ftransparent = False

    Config = namedtuple("Config", ["version", "awidth",
                                   "aheight", "fdpi",
                                   "ftransparent", "imgext"])
    config = Config(version, awidth, aheight, fdpi,
                    ftransparent, imgext)

    # Baseline
    script_paths = get_script_paths(config)
    logger = get_logger(config, script_paths)

    # Data
    data = get_data(config, script_paths, logger)

    # Process
    process(config, script_paths, logger, data)

    logger.completed()
