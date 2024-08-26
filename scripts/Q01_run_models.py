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

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

from pyrezeq import approx, steady, benchmarks, models, slow

# Tool to read data from tests
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

import importlib.util
freader = froot / "tests" / "data_reader.py"
spec = importlib.util.spec_from_file_location("data_reader", freader)
data_reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_reader)

#----------------------------------------------------------------------
# @Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="Run the 4 models using different ODE resolution methods", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-d", "--debug", help="Debug mode", \
                    action="store_true", default=False)
args = parser.parse_args()

debug = args.debug

model_names = ["QR", "BCR", "GRP", "GRPM"]
methods = ["analytical", "radau", "quasoare5", "quasoare100"]

start_daily = "2010-01-01"
end_daily = "2022-12-31"

start_hourly = "2022-02-01"
end_hourly = "2022-03-31"

nsubdiv = 50000

#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
fout = froot / "outputs"
fout.mkdir(exist_ok=True)

#----------------------------------------------------------------------
# @Logging
#----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename, contextual=True)
LOGGER.log_dict(vars(args), "Command line arguments")

#----------------------------------------------------------------------
# @Process
#----------------------------------------------------------------------
results = []
for isite, siteid in enumerate(data_reader.SITEIDS):
    if debug and isite>1:
        break

    LOGGER.context = f"{siteid}"

    LOGGER.info("Load data")
    daily = data_reader.get_data(siteid, "daily").loc[start_daily:end_daily]
    hourly = data_reader.get_data(siteid, "hourly").loc[start_hourly:end_hourly]

    # Prepare data for models
    rain = daily.loc[:, "RAINFALL[mm/day]"]
    evap = daily.loc[:, "PET[mm/day]"]
    climate = np.column_stack([rain, evap])
    climate_eff = np.column_stack([(rain-evap).clip(0), (evap-rain).clip(0)])
    dparams = 100+100*np.arange(8)
    dtimestep = 1. # time step in days

    inflows = hourly.loc[:, "STREAMFLOW_UP[m3/sec]"].interpolate()
    outflows = hourly.loc[:, "STREAMFLOW_DOWN[m3/sec]"].interpolate()
    q0 = inflows.quantile(0.9)
    htimestep = 3600. # time step in seconds
    hparams = q0*htimestep*24*(0.25+0.25*np.arange(8)) # store q0 for 1/4 to 2 days

    # Run models
    for mname in model_names:
        LOGGER.info(f"Model {mname}")
        # Model setup
        routing = mname in ["QR", "BCR"]
        if routing:
            timestep = htimestep
            params = hparams
            # routing exponent
            nu = 2. if mname == "QR" else 6.
            s0 = 0.
        else:
            timestep = dtimestep
            params = dparams
            s0 = 1./2

        # Fluxes functions
        if routing:
            fluxes, dfluxes = benchmarks.nonlinrouting_fluxes_noscaling(nu)
        elif mname == "GRP":
            fluxes, dfluxes = benchmarks.gr4jprod_fluxes_noscaling()
        elif mname == "GRPM":
            fpr = lambda x: (1.-x**3*(10-15*x+6*x**2)) if x>0 else 1.
            fae = lambda x: -(16*(x-0.5)**5+0.5) if x<1. else 4.-5.*x
            fperc = lambda x: -eta**4/4.*x**7 if x>0 else 0.
            fgw = lambda x: -2*x/(1+10*x) if x>0 else -2*x
            fluxes = [fpr, fae, fperc, fgw]

            dfpr = lambda x: -30.*x**2+60.*x**3-30*x**4 if x>0 else 0.
            dfae = lambda x: -5.+40*x-120*x**2+160.*x**3-80*x**4 if x<1. else -5.
            dfperc = lambda x: -eta**4*7./4.*x**6 if x>0 else 0.
            dfgw = lambda x: -2./(1+10*x)**2 if x>0 else -2. -2. -2. -2.
            dfluxes = [dfpr, dfae, dfperc, dfgw]


        for method, (iparam, param) in prod(methods, enumerate(params)):
            LOGGER.info(f"Method {method} / param {iparam+1}/{len(params)}", ntab=1)

            if method == "analytical":
                if mname == "GRPM":
                    continue

                elif mname in ["QR", "BCR"]:
                    theta = param
                    sim = benchmarks.nonlinrouting(nsubdiv, timestep, theta, \
                                        nu, q0, s0, inflows)

                elif mname == "GRP":
                    X1 = param
                    sim = benchmarks.gr4jprod(nsubdiv, X1, s0, climate)

            elif method == "radau":
                slow.

            elif method.startswith("quasoare"):
                nalphas = int(re.sub("quasoare", method))
                alphas = np.linspace(0, 2., nalphas)
                amat, bmat, cmat = approx.quad_coefficient_matrix(fluxes, alphas)

        sys.exit()




        LOGGER.info(f"Running {mname}/{method}")

LOGGER.completed()

