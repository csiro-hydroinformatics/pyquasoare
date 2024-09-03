#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2024-08-26 11:08:09.382739
## Comment : Run the 4 models using different ODE resolution ode_methods
##
## ------------------------------


import sys, os, re, json, math
import argparse
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
import data_utils

import importlib
importlib.reload(hdf5_utils)
importlib.reload(data_utils)

#----------------------------------------------------------------------
# @Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description="Run the 4 models using different ODE resolution methods", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-t", "--taskid", help="Configuration ID (model/siteid)", \
                    type=int, required=True)
parser.add_argument("-d", "--debug", help="Debug mode", \
                    action="store_true", default=False)
parser.add_argument("-l", "--flogs", help="Log folder", \
                    type=str, default="")
args = parser.parse_args()

debug = args.debug
taskid = args.taskid
flogs = args.flogs

# Get model name and siteid
model_name, siteid = data_utils.get_config(taskid)

# List of ODE ode_methods
ode_methods = data_utils.ODE_METHODS
if debug:
    ode_methods = ["radau", "c_quasoare_3"]

start_daily = "2010-01-01"
end_daily = "2022-12-31"

start_hourly = "2022-02-01"
end_hourly = "2022-03-31"

nsubdiv = 100 #50000

nparams = 3 if debug else 20

#----------------------------------------------------------------------
# @Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent

fout = froot / "outputs" / "simulations"
fout.mkdir(exist_ok=True, parents=True)

#----------------------------------------------------------------------
# @Logging
#----------------------------------------------------------------------
basename = source_file.stem
if flogs == "":
    flogs = froot / "logs" / "rezeqrun"
    flogs.mkdir(exist_ok=True, parents=True)
else:
    flogs = Path(flogs)
    assert flogs.exists()

if debug:
    fout = flogs / "simulations"
    fout.mkdir(exist_ok=True)

flog = flogs / f"rezeqrun_TASK{taskid}.log"
LOGGER = iutils.get_logger(basename, contextual=True, flog=flog)
LOGGER.log_dict(vars(args), "Command line arguments")
LOGGER.info(f"nconfig: {len(data_utils.CONFIGS)}")
LOGGER.info(f"Model  : {model_name}")
LOGGER.info(f"Siteid : {siteid}")

#----------------------------------------------------------------------
# @Process
#----------------------------------------------------------------------
fres = fout / f"simulations_TASK{taskid}.hdf5"
cfilt = hdf5_utils.COMPRESSION_FILTER

with tables.open_file(fres, "w", title="ODE simulations", filters=cfilt) as h5:

    # Create h5 groups
    h5_groups = {ode_method: h5.create_group("/", ode_method) \
                                        for ode_method in ode_methods}

    LOGGER.context = f"{siteid}"
    LOGGER.info("Load data")
    daily = data_utils.get_data(siteid, "daily").loc[start_daily:end_daily]
    hourly = data_utils.get_data(siteid, "hourly").loc[start_hourly:end_hourly]

    # Prepare data for models
    rain = daily.loc[:, "RAINFALL[mm/day]"]
    evap = daily.loc[:, "PET[mm/day]"]
    climate = np.column_stack([rain, evap])

    inflows = hourly.loc[:, "STREAMFLOW_UP[m3/sec]"].interpolate()
    outflows = hourly.loc[:, "STREAMFLOW_DOWN[m3/sec]"].interpolate()
    # .. rescale inflow to match outflow volume
    inflows_rescaled = outflows.mean()/inflows.mean()*inflows
    q0 = inflows_rescaled.quantile(0.9)

    # Run models
    # .. model setup
    routing = model_name in ["QR", "BCR"]
    if routing:
        timestep = 3600. # time step in seconds
        # store q0 for 0.2 to 10 days
        params = q0*timestep*24*np.linspace(0.2, 10., nparams)

        # routing exponent
        nu = 2. if model_name == "QR" else 6.
        s0 = 0.
    else:
        params = np.linspace(100, 2550, nparams)
        timestep = 1. # time step in days
        s0 = 1./2

    # .. fluxes functions
    if routing:
        fluxes, dfluxes = benchmarks.nonlinrouting_fluxes_noscaling(nu)

    elif model_name == "GRP":
        fluxes, dfluxes = benchmarks.gr4jprod_fluxes_noscaling()

    elif model_name == "GRPM":
        fpr = lambda x: (1.-x**3*(10-15*x+6*x**2)) if x>0 else 1.
        fae = lambda x: -(16*(x-0.5)**5+0.5) if x<1. else 4.-5.*x
        fperc = lambda x: -0.1*x**7 if x>0 else 0.
        fgw = lambda x: -0.1*x/(1+10*x) if x>0 else -2*x
        fluxes = [fpr, fae, fperc, fgw]

        dfpr = lambda x: -30.*x**2+60.*x**3-30*x**4 if x>0 else 0.
        dfae = lambda x: -5.+40*x-120*x**2+160.*x**3-80*x**4 if x<1. else -5.
        dfperc = lambda x: -0.7*x**6 if x>0 else 0.
        dfgw = lambda x: -0.1/(1+10*x)**2 if x>0 else -2. -2. -2. -2.
        dfluxes = [dfpr, dfae, dfperc, dfgw]

    # Run model for each parameter
    for iparam, param in enumerate(params):
        # Prepare scaling depending on param
        if routing:
            theta = param
            time_index = hourly.index
            nval = len(inflows)
            scalings = np.column_stack([inflows_rescaled/theta, \
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

        # Loop over ODE ode_method
        LOGGER.info(f"Param {iparam+1}/{nparams}")
        for ode_method in ode_methods:
            LOGGER.info(f"{ode_method} - start", ntab=1)

            if ode_method == "analytical":
                # Quasi analytical method

                if model_name == "GRPM":
                    LOGGER.info(f"{ode_method} - no analytical sol, skip",\
                                        ntab=1)
                    continue

                elif model_name in ["QR", "BCR"]:
                    theta = param
                    tstart = time.time()
                    qo = benchmarks.nonlinrouting(nsubdiv, timestep, theta, \
                                        nu, q0, s0, inflows)
                    runtime = (time.time()-tstart)*1e3
                    sim = np.column_stack([inflows, qo])
                    nval = len(inflows)
                    niter = nsubdiv*np.ones(nval)
                    s1 = np.nan*niter

                elif model_name == "GRP":
                    X1 = param
                    tstart = time.time()
                    sim = benchmarks.gr4jprod(nsubdiv, X1, s0, climate)
                    runtime = (time.time()-tstart)*1e3
                    s1 = sim[:, 0]
                    sim = sim[:, 1:]
                    niter = nsubdiv*np.ones(len(climate))

                s1_min = np.nanmin(s1)/param
                s1_max = np.nanmax(s1)/param
                alpha_min, alpha_max = np.nan, np.nan

            elif ode_method in ["radau", "rk45"]:
                # Numerical solver
                m = "Radau" if ode_method=="radau" else "RK45"
                tstart = time.time()
                niter, s1, sim = slow.numerical_model(fluxes, dfluxes, \
                                                        scalings, s0, \
                                                        timestep, method=m)
                runtime = (time.time()-tstart)*1e3
                sim = param*np.abs(sim)/timestep
                s1_min = np.nanmin(s1)
                s1_max = np.nanmax(s1)
                s1 *= param
                alpha_min, alpha_max = np.nan, np.nan

            elif re.search("quasoare", ode_method):
                # first go at alphas
                alphas = np.linspace(0, 5., 500)
                amat, bmat, cmat = approx.quad_coefficient_matrix(fluxes, alphas)
                stdy = steady.quad_steady_scalings(alphas, scalings, \
                                                    amat, bmat, cmat)

                # second go at alphas
                nalphas = int(re.sub(".*_", "", ode_method))
                alpha_min = np.nanmin(stdy)
                alpha_max = np.nanmax(stdy)
                alphas = np.linspace(alpha_min, alpha_max, nalphas)
                # .. insert a 0 if needed
                if alpha_min>1e-5:
                    alphas = np.insert(alphas, 0, 0.)
                amat, bmat, cmat = approx.quad_coefficient_matrix(fluxes, alphas)

                # Store flux approximation
                xx = np.linspace(alpha_min, alpha_max, 500)
                fx = approx.quad_fun_from_matrix(alphas, amat, bmat, cmat, xx)
                nfluxes = amat.shape[1]
                dfx = np.empty(fx.shape[0], dtype=hdf5_utils.FLUXES_DTYPE)
                dfx["s"] = xx
                for iflux in range(4):
                    if iflux<nfluxes:
                        f = np.array([fluxes[iflux](x) for x in xx])
                        dfx[f"flux{iflux+1}_true"] = f
                        dfx[f"flux{iflux+1}_approx"] = fx[:, iflux]
                    else:
                        dfx[f"flux{iflux+1}_true"] = np.nan
                        dfx[f"flux{iflux+1}_approx"] = np.nan

                tname = f"S{siteid}_{re.sub('_', '', ode_method)}"+\
                            f"_fluxes{iparam:03d}"
                h5_group = h5_groups[ode_method]
                tb = hdf5_utils.store(h5, h5_group, tname, dfx, \
                                 hdf5_utils.FluxesDescription)
                hdf5_utils.addmeta(tb, \
                        reated=datetime.now(), \
                        model_name=model_name, \
                        siteid=siteid, \
                        ode_method=ode_method, \
                        iparam=iparam, \
                        param=param, \
                        alpha_min=alpha_min, \
                        alpha_max=alpha_max, \
                        s1_min=s1_min, \
                        s1_max=s1_max)

                # Run model
                quad_model = models.quad_model if ode_method.startswith("c")\
                                        else slow.quad_model

                tstart = time.time()
                try:
                    niter, s1, sim = quad_model(alphas, scalings, \
                                        amat, bmat, cmat, s0, timestep)
                    runtime = (time.time()-tstart)*1e3
                    sim = param*np.abs(sim)/timestep
                    s1_min = np.nanmin(s1)
                    s1_max = np.nanmax(s1)
                    s1 *= param
                except Exception as err:
                    LOGGER.error(f"Error in {ode_method} run")
                    LOGGER.error(str(err))
                    sim = None

            if not sim is None:
                # Store result
                LOGGER.info(f"{ode_method} - store", ntab=1)
                simdata = hdf5_utils.format_sim(time_index, sim, s1, niter)
                tname = f"S{siteid}_{re.sub('_', '', ode_method)}"+\
                            f"_sim{iparam:03d}"
                h5_group = h5_groups[ode_method]
                tb = hdf5_utils.store(h5, h5_group, tname, simdata, \
                                      hdf5_utils.SimDescription)
                hdf5_utils.addmeta(tb, \
                        created=datetime.now(), \
                        model_name=model_name, \
                        siteid=siteid, \
                        ode_method=ode_method, \
                        runtime=runtime, \
                        iparam=iparam, \
                        param=param, \
                        alpha_min=alpha_min, \
                        alpha_max=alpha_max, \
                        s1_min=s1_min, \
                        s1_max=s1_max, \
                        niter_mean=niter.mean(), \
                        nsubdiv=nsubdiv, \
                        timestep=timestep)

    LOGGER.info("Closing hdf5 file")

LOGGER.completed()

