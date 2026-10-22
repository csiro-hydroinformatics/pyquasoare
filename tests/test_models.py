from pathlib import Path
import inspect
from itertools import product as prod
import time
import math
import re
import pytest

import numpy as np
import pandas as pd
import scipy.integrate as sci_integrate

from hydrodiy.io import iutils, csv

from pyquasoare import approx, models, slow, benchmarks, integrate, steady

from test_approx import generate_samples, reservoir_function

from pygme.models import gr4j

import data_reader

from loggers import get_logger

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent
LOGGER = get_logger(False, "models")

def test_quad_model_errors(allclose):
    nalphas = 20
    alphas = np.linspace(0., 1.2, nalphas)
    start, end = "2017-01", "2022-12-31"
    X1 = 200
    LOGGER.info("")

    # Compute approx coefs and introduce discontinuities
    # (i.e. should generate errors)
    fluxes, _ = benchmarks.gr4jprod_fluxes_noscaling()
    coefs = approx.quad_coefficient_matrix(fluxes, alphas)
    coefs[0, :, 0] = np.random.uniform(-10, 10, coefs.shape[1])

    # Get scaling data
    siteid = data_reader.SITEIDS[0]
    df = data_reader.get_data(siteid, "daily")
    df = df.loc[start:end]
    nval = len(df)
    cols = ["RAINFALL[mm_day-1]", "PET[mm_day-1]"]
    inputs = np.ascontiguousarray(df.loc[:, cols])
    climdiff = inputs[:, 0]-inputs[:, 1]
    scalings = np.column_stack([np.maximum(climdiff, 0.)/X1,
                                np.maximum(-climdiff, 0.)/X1,
                                np.ones(nval)])

    # Run model without raising
    s0 = 1./2
    niter, s1, fx = models.quad_model(alphas, scalings, coefs,
                                      s0, 1.)
    ierr = niter<0
    assert np.any(ierr)
    assert np.all(np.isnan(fx[ierr]))

    # Trigger warnings
    with pytest.warns(Warning):
        niter, s1, fx = models.quad_model(alphas, scalings, coefs,
                                          s0, 1.,
                                          errors="warn")

    # Trigger error
    with pytest.raises(ValueError, match="continuous"):
        niter, s1, fx = models.quad_model(alphas, scalings,
                                          coefs, s0, 1.,
                                          errors="raise")


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("X1", [50, 200, 1000])
def test_gr4jprod(siteid, X1, allclose):
    nalphas = 20
    alphas = np.linspace(0., 1.2, nalphas)
    start, end = "2017-01", "2022-12-31"
    nsubdiv = 50000
    LOGGER.info("")

    # Compute approx coefs
    fluxes, _ = benchmarks.gr4jprod_fluxes_noscaling()
    coefs = approx.quad_coefficient_matrix(fluxes, alphas)

    # Get climate data
    df = data_reader.get_data(siteid, "daily")
    df = df.loc[start:end]
    nval = len(df)
    cols = ["RAINFALL[mm_day-1]", "PET[mm_day-1]"]
    inputs = np.ascontiguousarray(df.loc[:, cols])
    climdiff = inputs[:, 0]-inputs[:, 1]
    errmax_max = 0.
    errbalmax_max = 0.

    scalings = np.column_stack([np.maximum(climdiff, 0.)/X1,
                                np.maximum(-climdiff, 0.)/X1,
                                np.ones(nval)])

    # Run approximate solution + exclude PR and AE
    s0 = X1/2
    expected = benchmarks.gr4jprod(nsubdiv, X1, s0, inputs)[:, :-2]

    # Run quad model
    s0 = 1./2
    niter, s1, fx = models.quad_model(alphas, scalings,
                                      coefs, s0, 1.)
    sims = np.column_stack([s1 * X1,
                            fx[:, 0] * X1,
                            -fx[:, 1] * X1,
                            -fx[:, 2] * X1])

    # Compare with slow
    niter_slow, s1_slow, fx_slow = slow.quad_model(alphas, scalings,
                                                   coefs, s0, 1.)
    sims_slow = np.column_stack([s1_slow * X1,
                                 fx_slow[:, 0] * X1,
                                -fx_slow[:, 1] * X1,
                                -fx_slow[:, 2] * X1])
    assert allclose(sims, sims_slow, atol=1e-5)

    # Error analysis
    ps = expected[:, 1]
    errbalmax = np.abs((sims[:, 1]-ps).mean(axis=0)) * 100
    errbalmax /= np.mean(ps)
    errbalmax_max = max(errbalmax, errbalmax_max)

    tol = 5e-4
    tsims = np.asinh(sims)
    texp = np.asinh(expected)
    errmax = np.abs(tsims - texp).max()
    msg = "PASS" if errmax < tol else "FAIL"
    mess = f"quad_model vs gr4jprod / {siteid} X1={X1:0.0f}:"\
                +f" err={errmax:0.2e}"\
                +f" errbal={errbalmax_max:0.2e}% > {msg}"
    LOGGER.info(mess)
    assert allclose(tsims, texp, atol=tol), mess


def test_quad_model_perturb(allclose):
    nalphas = 10
    alphas = np.linspace(0., 1.2, nalphas)
    start, end = "2017-01", "2022-12-31"
    X1 = 400

    # Compute approx coefs
    fluxes, _ = benchmarks.gr4jprod_fluxes_noscaling()
    coefs = approx.quad_coefficient_matrix(fluxes, alphas)

    # Get climate data
    siteid = data_reader.SITEIDS[0]
    df = data_reader.get_data(siteid, "daily")
    df = df.loc[start:end]
    nval = len(df)
    inputs = np.ascontiguousarray(df.loc[:, ["RAINFALL[mm_day-1]", "PET[mm_day-1]"]])
    climdiff = inputs[:, 0]-inputs[:, 1]
    scalings = np.column_stack([np.maximum(climdiff, 0.)/X1,
                                np.maximum(-climdiff, 0.)/X1,
                                np.ones(nval)])

    s0 = 1./2
    niter, s1, fx = models.quad_model(alphas, scalings, coefs,
                                      s0, 1.)

    perturb = 1e-10 * np.random.uniform(-1, 1, len(scalings))
    _, s1p, fxp = models.quad_model(alphas, scalings,
                                    coefs, s0, 1.,
                                    perturb=perturb)
    assert allclose(s1, s1p)
    assert allclose(fx, fxp)

    perturb = 1e-1 * np.random.uniform(-1, 1, len(scalings))
    _, s1p, fxp = models.quad_model(alphas, scalings,
                                    coefs, s0, 1.,
                                    perturb=perturb)
    assert np.abs(s1 - s1p).max() > 0.1


def test_quad_model_states_min_max(allclose):
    nalphas = 10
    alphas = np.linspace(0., 1.2, nalphas)
    start, end = "2017-01", "2022-12-31"
    X1 = 400

    # Compute approx coefs
    fluxes, _ = benchmarks.gr4jprod_fluxes_noscaling()
    coefs = approx.quad_coefficient_matrix(fluxes, alphas)

    # Get climate data
    siteid = data_reader.SITEIDS[0]
    df = data_reader.get_data(siteid, "daily")
    df = df.loc[start:end]
    nval = len(df)
    inputs = np.ascontiguousarray(df.loc[:, ["RAINFALL[mm_day-1]", "PET[mm_day-1]"]])
    climdiff = inputs[:, 0]-inputs[:, 1]
    scalings = np.column_stack([np.maximum(climdiff, 0.)/X1,
                                np.maximum(-climdiff, 0.)/X1,
                                np.ones(nval)])

    s0 = 1./2
    niter, s1, fx = models.quad_model(alphas, scalings, coefs,
                                      s0, 1.)
    assert s1.min() < 0.2
    assert s1.max() > 0.8

    smin, smax = 0.5, 0.51
    niter, s1c, fx = models.quad_model(alphas, scalings,
                                       coefs, s0, 1.,
                                    smin=smin, smax=smax)
    assert np.all(s1c >= smin)
    assert np.all(s1c <= smax)


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
    coefs = approx.quad_coefficient_matrix(fluxes, alphas)
    s0 = 0.
    niter, s1, sim = models.quad_model(alphas, scalings, coefs,
                                       s0, timestep)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("theta", [0.001, 0.01, 0.1])
def test_reservoir_interception(siteid, theta, allclose):
    df = data_reader.get_data(siteid, "daily")
    P, E = df.loc[:, ["RAINFALL[mm_day-1]", "PET[mm_day-1]"]].values[:300].T

    scalings = np.column_stack([P / theta, E / theta])

    nuP, nuE = 10., 10.
    fP = lambda x: 1 - np.minimum(x, 1.)**nuP
    fE = lambda x: -1 + (1 - np.maximum(x, 0.))**nuE

    nalphas = 50
    alphas = np.concatenate([[-1e-4], np.linspace(0., 1., nalphas - 2),
                             [1 + 1e-4]])
    coefs = approx.quad_coefficient_matrix([fP, fE], alphas)

    s0 = 0.
    timestep = 1.
    niter, s1, sim = models.quad_model(alphas, scalings, coefs,
                                       s0, timestep)
    Pr = P - sim[:, 0] * theta
    Er = E + sim[:, 1] * theta

    Pr_exp = np.maximum(P - E, 0.)
    tol = theta
    assert allclose(Pr, Pr_exp, atol=tol)

    Er_exp = np.maximum(E - P, 0.)
    assert allclose(Er, Er_exp, atol=tol)



def test_in_place_computing(allclose):
    siteid = "203005"
    start_hourly = "2022-02-01"
    end_hourly = "2022-03-31"
    hourly = data_reader.get_data(siteid, "hourly").loc[start_hourly:end_hourly]
    inflows = hourly.loc[:, "STREAMFLOW_UP[m3/sec]"].interpolate()

    q0 = inflows.quantile(0.9)
    timestep = 3600.
    nval = len(inflows)

    # Bi-Cubic reservoir
    nu = 6.
    fluxes, dfluxes = benchmarks.nonlinrouting_fluxes_noscaling(nu)
    nalphas = 5
    alphas = np.linspace(0, 3., nalphas)
    coefs = approx.quad_coefficient_matrix(fluxes, alphas)

    # Intialise two scalings
    scalings_ref = np.column_stack([inflows, q0 * np.ones(nval)])

    # Initialise outputs
    niter = np.zeros(nval, dtype=np.int32)
    s1 = np.zeros(nval)
    fx = np.zeros((nval, 2))

    nsims = 5
    sims = np.zeros((nval, nsims))
    s0 = 0.

    for ith, theta in enumerate(np.linspace(5e6, 5e7, nsims)):
        # Adjust scalings depending on paramter values
        scalings = scalings_ref / theta

        # Run model in place (i.e. no new array allocation)
        models.quad_model(alphas, scalings, coefs,
                          s0, timestep,
                          niter=niter, fluxes=fx, s1=s1)
        sims[:, ith] = -fx[:, 1] * theta / timestep

        scalings *= theta

    # Check that simulations are different
    assert (sims.std(axis=1) > 10).sum() > 100
