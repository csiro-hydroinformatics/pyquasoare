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

methods = ["analytical", "radau", "rk45", \
                "py_quasoare_5", "py_quasoare_100", \
                "c_quasoare_5", "c_quasoare_100"]
methods = ["rk45"]

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
# HDF5 simulation table structure
class Simulation(tables.IsDescription):
    flux1 = tables.Float64Col()
    flux2 = tables.Float64Col()
    flux3 = tables.Float64Col()
    flux4 = tables.Float64Col()
    s1 = tables.Float64Col()
    niter = tables.Int32Col()


fres = fout / "simulations.hdf5"
with tables.open_file(fres, "w", title="ODE simulations") as h5:
    # Create a group by model
    groups = {}
    for model_name in model_names:
        groups[model_name] = h5.create_group("/", model_name)

    # Run models for each site
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
        dparams = 100+100*np.arange(8)
        dtimestep = 1. # time step in days

        inflows = hourly.loc[:, "STREAMFLOW_UP[m3/sec]"].interpolate()
        outflows = hourly.loc[:, "STREAMFLOW_DOWN[m3/sec]"].interpolate()
        q0 = inflows.quantile(0.9)
        htimestep = 3600. # time step in seconds
        hparams = q0*htimestep*24*(0.25+0.25*np.arange(8)) # store q0 for 1/4 to 2 days

        # Run models
        for model_name in model_names:
            LOGGER.info(f"Model {model_name}")

            # Model setup
            routing = model_name in ["QR", "BCR"]
            if routing:
                timestep = htimestep
                params = hparams
                # routing exponent
                nu = 2. if model_name == "QR" else 6.
                s0 = 0.
            else:
                timestep = dtimestep
                params = dparams
                s0 = 1./2

            # Fluxes functions
            if routing:
                fluxes, dfluxes = benchmarks.nonlinrouting_fluxes_noscaling(nu)

            elif model_name == "GRP":
                fluxes, dfluxes = benchmarks.gr4jprod_fluxes_noscaling()

            elif model_name == "GRPM":
                eta = 1./2.25
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

            # Run models
            for iparam, param in enumerate(params):
                # Prepare scaling depending on param
                if routing:
                    theta = param
                    time_index = hourly.index
                    nval = len(inflows)
                    scalings = np.column_stack([inflows/theta, \
                                            q0/theta*np.ones(nval)])
                elif model_name == "GRP":
                    X1 = param
                    time_index = daily.index
                    ones = np.ones(len(climate))
                    scalings = np.column_stack([np.maximum(rain-evap, 0)/X1, \
                                            np.maximum(evap-rain, 0)/X1, ones])
                elif model_name == "GRPM":
                    X1 = param
                    time_index = daily.index
                    ones = np.ones(len(climate))
                    scalings = np.column_stack([np.maximum(rain-evap, 0)/X1, \
                                            np.maximum(evap-rain, 0)/X1, ones, ones])

                # Loop over ODE method
                for method in methods:
                    LOGGER.info(f"Method {method} / param"\
                                    +f" {iparam+1}/{len(params)}", ntab=1)
                    if method == "analytical":
                        if model_name == "GRPM":
                            continue

                        elif model_name in ["QR", "BCR"]:
                            theta = param
                            tstart = time.time()
                            qo = benchmarks.nonlinrouting(nsubdiv, timestep, theta, \
                                                nu, q0, s0, inflows)
                            runtime = (time.time()-tstart)*1e3
                            sim = np.column_stack([inflows, qo])
                            nval = len(inflows)
                            niter = np.ones(nval)
                            s1 = np.nan*niter

                        elif model_name == "GRP":
                            X1 = param
                            tstart = time.time()
                            sim = benchmarks.gr4jprod(nsubdiv, X1, s0, climate)
                            runtime = (time.time()-tstart)*1e3
                            s1 = sim[:, 0]
                            sim = sim[:, 1:]
                            niter = np.ones(len(climate))

                    elif method in ["radau", "rk45"]:
                        m = "Radau" if method=="radau" else "RK45"
                        tstart = time.time()
                        niter, s1, sim = slow.numerical_model(fluxes, dfluxes, \
                                                                scalings, s0, \
                                                                timestep, method=m)
                        runtime = (time.time()-tstart)*1e3
                        sim = param*np.abs(sim)/timestep
                        s1 *= param

                    elif re.search("quasoare", method):
                        nalphas = int(re.sub(".*_", "", method))
                        alpha_max = 1.2 if model_name.startswith("GR") else 3.
                        alphas = np.linspace(0, alpha_max, nalphas)
                        amat, bmat, cmat = approx.quad_coefficient_matrix(fluxes, alphas)

                        quad_model = models.quad_model if method.startswith("c")\
                                                else slow.quad_model

                        tstart = time.time()
                        niter, s1, sim = quad_model(alphas, scalings, \
                                                amat, bmat, cmat, s0, timestep)
                        runtime = (time.time()-tstart)*1e3
                        sim = param*np.abs(sim)/timestep
                        s1 *= param

                    # Store result
                    gr = groups[model_name]
                    tname = f"{siteid}_{re.sub('_', '', method)}_param{iparam}"
                    simdata = np.column_stack([sim, s1, niter])
                    ar = h5.create_array(gr, tname, simdata,\
                                            "ODE simulation",\
                                            atom=tables.Float64Atom(),\
                                            shape=simdata.shape)
                    ar.attrs.created = datetime.now()
                    ar.attrs.ode_method = method
                    ar.attrs.model_name = model_name
                    ar.attrs.runtime_ms = runtime

LOGGER.completed()

