#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2024-08-26 11:08:09.382739
## Comment : Run the 4 models using different ODE resolution methods
##
## ------------------------------


import sys, os, re, json, math
import argparse
from itertools import product as prod
from pathlib import Path
import time
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tables

from hydrodiy.io import csv, iutils

from pyrezeq import approx, steady, benchmarks, models, slow
import hdf5_utils

# Tool to read data from tests
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

import importlib
freader = froot / "tests" / "data_reader.py"
spec = importlib.util.spec_from_file_location("data_reader", freader)
data_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_reader)

importlib.reload(hdf5_utils)

#----------------------------------------------------------------------
# @Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="Compute performance of ODE methods", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--debug", help="Debug mode", \
                    action="store_true", default=False)
args = parser.parse_args()

debug = args.debug

#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
fdata = froot / "outputs" / "simulations"
fout = fdata.parent

#----------------------------------------------------------------------
# @Logging
#----------------------------------------------------------------------
basename = source_file.stem
flogs = froot / "logs"
flogs.mkdir(exist_ok=True)
flog = flogs / f"{source_file.stem}.log"
LOGGER = iutils.get_logger(basename, flog=flog)
LOGGER.log_dict(vars(args), "Command line arguments")

#----------------------------------------------------------------------
# @Process
#----------------------------------------------------------------------
lf = list(fdata.glob("*.hdf5"))
nfiles = len(lf)
all_results = []
for ifile, f in enumerate(lf):
    LOGGER.info(f"Processing file {f.stem} - {ifile+1}/{nfiles}")
    with tables.open_file(f, "r") as h5:
        sims = {}
        fluxes = {}
        results = []
        for group in h5.root:
            for tb in h5.list_nodes(group, "Leaf"):
                info = {}
                cc = ["ode_method", "siteid", "model_name", \
                            "param", "iparam", "runtime", \
                            "timestep", "niter_mean", \
                            "alpha_min", "s1_min", \
                            "alpha_max", "s1_max"]
                attrs = tb.attrs
                for cn in cc:
                    if hasattr(attrs, cn):
                        info[cn] = getattr(attrs, cn)

                df = hdf5_utils.convert(tb[:])
                method, iparam = info["ode_method"], info["iparam"]
                if re.search("sim", tb.name):
                    sims[(method, iparam)] = df
                    info["ntimestep"] = len(df)
                    nalphas = int(re.sub(".*quasoare_", "", method))\
                                if re.search("quasoare_", method) else 0
                    info["nalphas"] = nalphas
                    results.append(info)
                else:
                    fluxes[(method, iparam)] = df

        results = pd.DataFrame(results)

        # Compute perf
        nparams = results.iparam.max()+1
        methods = results.ode_method.unique()

        cabs_sim = [f"ERRABSMAX_SIM_FLUX{f}" for f in range(1, 5)]
        cbal = [f"ERRBAL_SIM_FLUX{f}[%]" for f in range(1, 5)]
        cabs_interp = [f"ERRABSMAX_INTERP_FLUX{f}" for f in range(1, 5)]
        crunt = "RUNTIME_RATIO[%]"
        cniter = "NITER_RATIO[%]"
        cc = cabs_sim+cbal+cabs_interp+[crunt, cniter]
        results.loc[:, cc] = np.nan
        model_names = results.loc[:, "model_name"].unique()

        for iparam, model_name in prod(range(nparams), model_names):
            # Define reference results
            method_ref = "analytical" if model_name=="QR" else "radau"
            ref = sims[(method_ref, iparam)].filter(regex="flux", axis=1)

            idx = (results.ode_method == method_ref) \
                    & (results.iparam==iparam)\
                    & (results.model_name==model_name)

            ref_runt = results.loc[idx, "runtime"].squeeze()
            ref_niter = results.loc[idx, "niter_mean"].squeeze()

            # Compute perf for each method
            for method in methods:
                if iparam == 0:
                    LOGGER.info(f"perf for {method}", ntab=1)

                if method == method_ref:
                    continue

                idx = (results.ode_method == method)\
                        & (results.iparam==iparam)\
                        & (results.model_name==model_name)

                sim = sims[(method, iparam)].filter(regex="flux", axis=1)

                eabsmax = np.abs(sim-ref).max()
                results.loc[idx, cabs_sim] = eabsmax.values

                ebal = np.abs(np.sum(sim-ref, axis=0))\
                            /np.abs(np.sum(ref, axis=0))*100
                results.loc[idx, cbal] = ebal.values

                runt = results.loc[idx, "runtime"].squeeze()
                results.loc[idx, crunt] = runt/ref_runt*100

                niter = results.loc[idx, "niter_mean"].squeeze()
                results.loc[idx, cniter] = niter/ref_niter*100

                # flux approx
                key = (method, iparam)
                if key in fluxes:
                    fx = fluxes[key]
                    ft = fx.filter(regex="_true", axis=1)
                    ft.columns = [re.sub("_.*", "", cn) for cn in ft.columns]
                    fa = fx.filter(regex="_approx", axis=1)
                    fa.columns = [re.sub("_.*", "", cn) for cn in fa.columns]
                    results.loc[idx, cabs_interp] = np.abs(fa-ft).max().values

    all_results.append(results)

fr = fout / "results.csv"
all_results = pd.concat(all_results)
csv.write_csv(all_results, fr, "Concatenated results", \
                source_file, compress=False, lineterminator="\n", \
                float_format="%0.8e")

LOGGER.completed()

