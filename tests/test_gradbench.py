from pathlib import Path
from itertools import product as prod
import time
import math
import re
import pytest

import numpy as np
import pandas as pd

from hydrodiy.io import iutils

from pyquasoare import approx, models, gradbench, steady

import data_reader

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent

MODELS = {
    1: "quadratic routing",
    2: "simplified GR4J production",
    3: "simplified GR4J routing"
    }

def get_data(siteid):
    df = data_reader.get_data(siteid, "daily")
    df = df.interpolate()
    pin = df.iloc[:, 0].values
    ein = df.iloc[:, 1].values
    qin = df.iloc[:, 2].values
    x0 = 0.5
    timestep = 1.
    return x0, timestep, qin, pin, ein


def get_coefs(model_id, amax):
    # quad routing
    if model_id == 1:
        fluxes = [
            lambda x: np.ones_like(x),
            lambda x: np.zeros_like(x),
            lambda x: x**2
            ]

    # simplified rr production
    elif model_id == 2:
        fluxes = [
            lambda x: 1 - x**2,
            lambda x: x * (2 - x),
            lambda x: 4**3 / 9**4 * x**2
            ]

    # simplified rr routing
    elif model_id == 3:
        fluxes = [
            lambda x: 0.9 * np.ones_like(x),
            lambda x: x**2,
            lambda x: x**2 / 4
            ]

    alphas = np.linspace(0, amax, 3)
    return alphas, approx.quad_coefficient_matrix(fluxes, alphas, 2)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("gamma", [10, 100, 10000])
@pytest.mark.parametrize("timestep", [1., 0.1])
def test_gradbench_quadrouting(siteid, gamma, timestep, allclose):
    x0, _, qin, _, _ = get_data(siteid)

    # run gradbench
    gammas = np.array([gamma])
    states, qout = gradbench.gradbench_quadrouting(timestep, qin, x0, gammas)
    qout = qout.squeeze()
    states = states.squeeze()

    # Run quasoare
    amax = (1 + round(states.max() / 0.2)) * 0.2
    alphas, coefs = get_coefs(1, amax)
    nval = len(qin)
    scalings = np.ones((nval, 3))
    scalings[:, 0] = qin / gamma
    scalings[:, 1] = 0.
    scalings[:, 2] = -1. / timestep

    _, x1, fx = models.quad_model(alphas, scalings, coefs, x0,
                                  timestep=timestep)
    expected = -fx[:, -1] * gamma / timestep

    assert allclose(states, x1)
    assert allclose(qout, expected)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("gamma", [10, 100, 10000])
@pytest.mark.parametrize("timestep", [1., 0.1])
def test_gradbench_rainfallrunoff_prod(siteid, gamma, timestep, allclose):
    x0, _, _, pin, ein = get_data(siteid)

    # run gradbench
    gamma_rr_prod = gamma
    gamma_rr_route = gamma / 5.
    demand_rr_route = 0.
    out = gradbench.gradbench_rainfallrunoff(timestep, pin, ein, x0,
                                             gamma_rr_prod,
                                             gamma_rr_route,
                                             demand_rr_route)
    states_p, states_r, qout_p, qout_r = out

    # Run quasoare
    amax = (1 + round(states_p.max() / 0.2)) * 0.2
    alphas, coefs = get_coefs(2, amax)
    nval = len(pin)
    scalings = np.ones((nval, 3))
    scalings[:, 0] = pin / gamma_rr_prod
    scalings[:, 1] = -ein / gamma_rr_prod
    scalings[:, 2] = -1. / timestep
    _, x_p, fx_p = models.quad_model(alphas, scalings, coefs, x0,
                                     timestep=timestep)
    expected = pin - (fx_p[:, 0] + fx_p[:, 2]) * gamma_rr_prod / timestep

    assert allclose(states_p, x_p)
    assert allclose(qout_p, expected)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("gamma", [10, 100, 10000])
@pytest.mark.parametrize("demand", [-10., 0., 10.])
def test_gradbench_rainfallrunoff_route(siteid, gamma, demand, allclose):
    pytest.skip("WIP")
    x0, timestep, qin, pin, ein = get_data(siteid)

    # run gradbench
    gamma_rr_prod = gamma
    gamma_rr_route = gamma / 5.
    demand_rr_route = demand
    out = gradbench.gradbench_rainfallrunoff(timestep, pin, ein, x0,
                                             gamma_rr_prod,
                                             gamma_rr_route,
                                             demand_rr_route)
    states_p, states_r, qout_p, qout_r = out

    # Run quasoare
    amax = (1 + round(states_r.max() / 0.2)) * 0.2
    alphas, coefs = get_coefs(3, amax)
    nval = len(pin)
    scalings = np.ones((nval, 3))
    scalings[:, 0] = qout_p / gamma_rr_route
    scalings[:, 1] = demand / gamma_rr_route
    scalings[:, 2] = -1
    _, x_r, fx_r = models.quad_model(alphas, scalings, coefs, x0,
                                     timestep=timestep)
    expected = -fx_r[:, 2] * gamma_rr_route

    assert allclose(states_r, x_r)
    assert allclose(qout_r, expected)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("gamma", [1, 100, 10000])
def test_gradbench_quad_routing_multi_reach(siteid, gamma, allclose):
    pytest.skip("WIP")
    model_id = 1
    x0, timestep, qin, pin, ein = get_data(siteid)
    nreaches = 5

    # run gradbench
    gammas = np.array([gamma] * nreaches)
    states, qout = gradbench.gradbench_quadrouting(timestep, qin, x0, gammas)

    # Run quasoare
    amax = (1 + round(states.max() / 0.2)) * 0.2
    alphas, coefs = get_coefs(model_id, amax)

    sexpected = np.zeros_like(states)
    qexpected = np.zeros_like(qout)
    for i in range(nreaches):
        qup = qin if i == 0 else qexpected[:, i - 1]
        scalings = get_scalings(qup, qup * 0., qup * 0., gamma, model_id, 0.)
        _, x1, fx = models.quad_model(alphas, scalings, coefs, x0,
                                      timestep=timestep)
        qexpected[:, i] = -fx[:, -1] * gamma
        sexpected[:, i] = x1

    assert allclose(states, sexpected)
    assert allclose(qout, qexpected)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("gamma", [1, 100, 10000])
def test_gradbench_tangent(siteid, gamma, allclose):
    return
    x0, timestep, qin = get_data(siteid)
    eps = 1e-6

    # run gradbench
    gammas = np.array([gamma - eps])
    states_1, qout_1 = gradbench.gradbench(timestep, qin, x0, gammas)
    qout_1 = qout_1.squeeze()
    states_1 = states_1.squeeze()

    gammas = np.array([gamma + eps])
    states_2, qout_2 = gradbench.gradbench(timestep, qin, x0, gammas)
    qout_2 = qout_2.squeeze()
    states_2 = states_2.squeeze()

    sexpected = (states_2 - states_1) / 2 / eps
    qexpected = (qout_2 - qout_1) / 2 / eps

    # run gradbench tangent
    gammas = np.array([gamma])
    gammas_diff = np.array([1.])
    states, qout, states_diff, qout_diff = gradbench.gradbench_tangent(timestep, qin, x0,
                                                                     gammas,
                                                                     gammas_diff)
    assert allclose(sexpected, states_diff.squeeze())
    assert allclose(qexpected, qout_diff.squeeze(), atol=5e-7)
