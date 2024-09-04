from pathlib import Path
from itertools import product as prod
import time
import math
import re
import pytest

import numpy as np
import pandas as pd
import scipy.integrate as sci_integrate

from hydrodiy.io import iutils

from pyrezeq import approx, models, slow, benchmarks, integrate

from test_approx import generate_samples, reservoir_function

from pygme.models import gr4j

import data_reader

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent
LOGGER = iutils.get_logger("models", flog=FTEST / "test_models.log")

def test_quad_model(allclose):
    nalphas = 20
    alphas = np.linspace(0., 1.2, nalphas)
    start, end = "2017-01", "2022-12-31"
    nsubdiv = 50000
    X1s = [50, 200, 1000]
    LOGGER.info("")

    # Compute approx coefs
    fluxes, _ = benchmarks.gr4jprod_fluxes_noscaling()
    amat, bmat, cmat = approx.quad_coefficient_matrix(fluxes, alphas)

    # Loop over sites
    for isite, siteid in enumerate(data_reader.SITEIDS):
        # Get climate data
        df = data_reader.get_data(siteid, "daily")
        df = df.loc[start:end]
        nval = len(df)
        inputs = np.ascontiguousarray(df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]])
        climdiff = inputs[:, 0]-inputs[:, 1]
        errmax_max = 0.
        errbalmax_max = 0.
        for X1 in X1s:
            scalings = np.column_stack([np.maximum(climdiff, 0.)/X1, \
                                        np.maximum(-climdiff, 0.)/X1, \
                                        np.ones(nval)])

            # Run approximate solution + exclude PR and AE
            s0 = X1/2
            expected = benchmarks.gr4jprod(nsubdiv, X1, s0, inputs)[:, :-2]

            # Run quad model
            s0 = 1./2
            niter, s1, fx = models.quad_model(alphas, scalings, \
                                            amat, bmat, cmat, s0, 1.)

            sims = np.column_stack([s1*X1, fx[:, 0]*X1, \
                                        -fx[:, 1]*X1, -fx[:, 2]*X1])

            # Compare with slow
            niter_slow, s1_slow, fx_slow = slow.quad_model(alphas, scalings, \
                                            amat, bmat, cmat, s0, 1.)

            sims_slow = np.column_stack([s1_slow*X1, fx_slow[:, 0]*X1, \
                                        -fx_slow[:, 1]*X1, -fx_slow[:, 2]*X1])
            assert allclose(sims, sims_slow, atol=1e-5)

            # Error analysis
            errmax = np.abs(sims-expected).max()
            errmax_max = max(errmax, errmax_max)

            ps = expected[:, 1]
            errbalmax = np.abs((sims[:, 1]-ps).mean(axis=0))*100
            errbalmax /= np.mean(ps)
            errbalmax_max = max(errbalmax, errbalmax_max)

            atol = 1e-3
            rtol = 1e-4
            assert allclose(expected, sims, atol=atol, rtol=rtol)

        mess = f"quad_model vs gr4jprod / site {isite+1}:"\
                    +f" errmax={errmax_max:3.3e}"\
                    +f" errbalmax={errbalmax_max:3.3e}%"
        LOGGER.info(mess)


def test_routing_convergence(allclose):
    siteid = "203005"
    start_hourly = "2022-02-01"
    end_hourly = "2022-03-31"
    hourly = data_reader.get_data(siteid, "hourly").loc[start_hourly:end_hourly]
    inflows = hourly.loc[:, "STREAMFLOW_UP[m3/sec]"].interpolate()

    q0 = inflows.quantile(0.9)
    theta = 1173678.55
    timestep = 3600.
    nval = len(inflows)
    scalings = np.column_stack([inflows/theta, \
                                 q0/theta*np.ones(nval)])
    nu = 2.
    fluxes, dfluxes = benchmarks.nonlinrouting_fluxes_noscaling(nu)

    nalphas = 5
    alphas = np.linspace(0, 3., nalphas)
    amat, bmat, cmat = approx.quad_coefficient_matrix(fluxes, alphas)
    s0 = 0.
    niter, s1, sim = models.quad_model(alphas, scalings, \
                            amat, bmat, cmat, s0, timestep)


def test_bicubic(allclose):
    # Configure integration
    timestep = 3600
    theta = 40119539.59355378
    fluxes, dfluxes = benchmarks.nonlinrouting_fluxes_noscaling(nu=6)
    s0 = 0.

    # Get data
    siteid = "203014"
    start_hourly = "2022-02-01"
    end_hourly = "2022-03-03"
    hourly = data_reader.get_data(siteid, "hourly").loc[start_hourly:end_hourly]
    inflows = hourly.loc[:, "STREAMFLOW_UP[m3/sec]"].interpolate()
    outflows = hourly.loc[:, "STREAMFLOW_DOWN[m3/sec]"].interpolate()

    inflows_rescaled = outflows.mean()/inflows.mean()*inflows
    q0 = inflows_rescaled.quantile(0.9)
    nval = len(inflows)
    scalings = np.column_stack([inflows_rescaled/theta, \
                            q0/theta*np.ones(nval)])

    # Numerical
    nitera, s1a, fxa = slow.numerical_model(fluxes, dfluxes, \
                                           scalings, s0, \
                                           timestep, method="Radau")

    # Quasoare
    alphas = np.linspace(0., 1.5, 500)
    amat, bmat, cmat = approx.quad_coefficient_matrix(fluxes, alphas)
    s_start = s0
    fxb = []
    for t in range(nval):
        n, s_end, f = slow.quad_integrate(alphas, scalings[t], \
                                        amat, bmat, cmat, 0., s_start, \
                                        timestep, debug=False)
        #assert allclose(f, fxa[t])
        fxb.append(f)
        s_start = s_end

    fxb = np.array(fxb)
    assert np.allclose(fxa, fxb)


    niterb, s1c, fxc = models.quad_model(alphas, scalings, amat, bmat, cmat, \
                                                    s0, timestep)
    assert np.allclose(fxa, fxc)

