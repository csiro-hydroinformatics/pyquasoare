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
from matplotlib import ticker as mticker

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils, violinplot

# Package to plot spatial data
from hydrodiy.io import csv, iutils

# Select backend
mpl.use("Agg")


def get_script_paths(config):
    source_file = Path(__file__).resolve()
    froot = source_file.parent.parent.parent
    fdata = froot / "outputs" / f"gradaverage_v{config.version}"
    fimg = froot / "images" / "manuscript" / source_file.stem
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
    nas = perf.TASK_OPT_nalphas.unique()
    nsmaxs = perf.TASK_OPT_nsubmax.unique()

    gm_ref = "500.0_50.0_500.0"
    na_ref = 50

    for na, gm in prod(nas, gms):
        logger.info(f"Plotting na={na} gm={gm}", nret=1)

        # Create figure
        plt.close("all")
        nrs = [1, 5, 20]
        mosaic = [["."] + [f"{nr}/title" for nr in nrs]] \
            + [[f"title/perf"] + [f"{nr}/perf" for nr in nrs]]

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

        idx0 = perf.TASK_OPT_nalphas == na
        idx0 &= perf.TASK_OPT_gamma_mean == gm

        for iax, (aname, ax) in enumerate(axs.items()):
            if re.search("title", aname):
                ax.axis("off")
                ax.set(xlim=(0, 1), ylim=(0, 1))
                t1, t2 = aname.split("/")
                if t2 == "title":
                    t1 = int(t1)
                    title = f"{t1} reach" if t1 == 1 else f"{t1} reaches"

                    idx = idx0 & (perf.TASK_OPT_nreaches == int(t1))
                    nr = t1 * 9
                    nv = (perf.loc[idx, "nvariables"].unique()).squeeze()
                    ny = (perf.loc[idx, "ntimesteps_agg"].unique() / 365.25).squeeze()
                    ng = nr * 6 * nv
                    subtitle = f"{nr:0.0f} Reservoirs  {nv:0.0f} Indep. vars." \
                        + f"  {ny:0.0f} Years\n" \
                        + f"({ng:0.0f} gradient time series)"
                    ax.text(0.5, 0.1, subtitle,
                            va="center", ha="center",
                            fontsize="large")
                    rot = 0.
                    va = "bottom"
                else:
                    title = "Average $E_{max}^{90}$ Error/stdev ratio\nabove 90th prc [-]"
                    rot = 90.
                    va = "center"

                ax.text(0.5, 0.5, title,
                        va=va, ha="center",
                        fontweight="bold",
                        fontsize="x-large",
                        rotation=rot)
                continue

            nreaches, _ = aname.split("/")
            nreaches = int(nreaches)
            idx = idx0 & (perf.TASK_OPT_nreaches == nreaches)
            logger.info(f"Plot reaches={nreaches}", ntab=1)
            if idx.sum() == 0:
                logger.info("No data. Skip.", ntab=2)
                continue

            metrics = [f"{vn}_{me}" for vn in ["fx", "dfx"]
                       for me in ["log10errmax00", "log10errmax90"]]
            nsmaxs = perf.TASK_OPT_nsubmax.unique()
            data = []
            for metric, nsmax in prod(metrics, nsmaxs):
                pat = "^" + re.sub("_", "_([^_]+|)", metric)
                cc = [cn for cn in perf.columns if re.search(pat, cn)]
                cc = [cn for cn in cc if not re.search(f"nsubsteps{nsmax:03d}", cn)]
                idxm = idx & (perf.TASK_OPT_nsubmax == nsmax)
                dd = perf.loc[idxm, cc].mean().reset_index()
                df = dd.loc[:, "index"].str[:-3].str.split("_", expand=True)
                df.columns = ["varname", "metric", "n"]
                df.loc[:, "nsub"] = df.n.str.replace("nsubsteps", "").astype(int)
                df.loc[:, "ts"] = 24/ df.nsub
                df.drop("n", axis=1, inplace=True)
                df.loc[:, "perf"] = dd.loc[:, 0]
                df.loc[:, "nsubmax"] = nsmax
                data.append(df)

            data = pd.concat(data)
            ii = data.metric == "log10errmax90"
            is_log = True
            data = data.loc[ii]

            kw = {"index": "ts", "columns":["nsubmax", "varname"],
                  "values": "perf"}
            check = pd.pivot_table(data, aggfunc=len, **kw)
            assert all(check == 1)

            data = pd.pivot_table(data, **kw)
            vns = set([cn[1] for cn in data.columns])
            cols = {v:None for v in vns}
            ns = set([cn[0] for cn in data.columns])
            ls = {v: "-" if v == 24 else ":" for v in ns}
            lw = {v: 4 if v == 24 else 2 for v in ns}
            cc = [(n, vn) for vn in vns for n in ns]
            data = data.loc[:, cc]

            for (ns, vn), se in data.items():
                ns = int(ns)
                vname = "Fluxes" if vn == "fx" else "Flux gradients"
                nname = f"{24*60 / ns:0.0f}min ref. substep"
                lab = f"{vname}\n{nname}"#
                m = "o" if ns == 24 else None
                ax.plot(se.index, se.values,
                        ls=ls[ns], color=cols[vn],
                        lw=lw[ns], marker=m,
                        label=lab)

                cols[vn] = ax.get_lines()[-1].get_color()

            # Axis
            title = f"({letters[iplot]})"
            ax.set_title(title, x=0.02, y=0.95,
                         va="top", ha="left",
                         fontsize="x-large")

            ax.set(xlabel="",
                   ylabel="")

            if is_log:
                ax.set_ylim((-3., 0.))
                tk = np.arange(-3, 1, 1)
                tkm = []
                for t in tk[:-1]:
                    tkm.append(np.log10(10.**t * np.arange(1, 10)))
                tkm = np.concatenate(tkm)
                ax.set_yticks(tkm, minor=True)

                labs = ["1" if k == 0 else f"$10^{{{k}}}$" for k in tk]
                ax.set_yticks(tk, labels=labs, minor=False)

            ax.grid(axis="y")
            ax.legend(loc=4)

            iplot += 1

        # Save file
        cfg = f"N{na}_G{gm}"
        fp = script_paths.fimg / \
                f"figH_gradaverage_v{config.version}_{cfg}.{config.imgext}"
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
    awidth = 3.5
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
