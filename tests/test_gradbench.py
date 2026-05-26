from pathlib import Path
from itertools import product as prod
import time
import math
import re
import pytest

import numpy as np
import pandas as pd

from hydrodiy.io import iutils

from pyquasoare import approx, models, benchmarks, steady

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

    # run benchmarks
    gammas = np.array([gamma])
    states, qout = benchmarks.gradbench_quadrouting(timestep, qin, x0, gammas)
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
def test_gradbench_quadrouting_tangent(siteid, gamma, timestep, allclose):
    x0, _, qin, _, _ = get_data(siteid)

    # run benchmarks
    gammas = np.array([gamma])
    dgammas = np.array([1.])
    out = benchmarks.gradbench_quadrouting_tangent(timestep, qin, x0,
                                                   gammas, dgammas)
    states, dstates, qout, dqout = out

    qout = qout.squeeze()
    dqout = dqout.squeeze()
    states = states.squeeze()
    dstates = dstates.squeeze()

    # Numerical differenciation
    eps = 1e-6
    gammas = np.array([gamma - eps])
    states1, qout1 = benchmarks.gradbench_quadrouting(timestep, qin, x0, gammas)
    gammas = np.array([gamma + eps])
    states2, qout2 = benchmarks.gradbench_quadrouting(timestep, qin, x0, gammas)

    dqout_expected = (qout2 - qout1).squeeze() / 2 / eps
    atol = 1e-5 if gamma == 10000 else 1e-8
    assert allclose(dqout, dqout_expected, atol=atol)

    atol = 1e-8
    dstates_expected = (states2 - states1).squeeze() / 2 / eps
    assert allclose(dstates, dstates_expected, atol=atol)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("gamma", [10, 100, 10000])
@pytest.mark.parametrize("timestep", [1., 0.1])
def test_gradbench_rainfall_runoff_prod(siteid, gamma, timestep, allclose):
    x0, _, _, pin, ein = get_data(siteid)

    # run benchmarks
    gamma_rr_prod = gamma
    gamma_rr_route = gamma / 5.
    demand_rr_route = 0.
    out = benchmarks.gradbench_rainfall_runoff(timestep, pin, ein, x0,
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
@pytest.mark.parametrize("demand", [-0.1, 0, 0.1])
@pytest.mark.parametrize("timestep", [1., 0.1])
def test_gradbench_rainfall_runoff_route(siteid, gamma, demand, timestep, allclose):
    x0, _, _, pin, ein = get_data(siteid)

    # run benchmarks
    gamma_rr_prod = gamma
    gamma_rr_route = gamma / 5.
    demand_rr_route = demand
    out = benchmarks.gradbench_rainfall_runoff(timestep, pin, ein, x0,
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
    scalings[:, 2] = -1 / timestep
    _, x_r, fx_r = models.quad_model(alphas, scalings, coefs, x0,
                                     timestep=timestep)
    expected = qout_p * 0.1 - fx_r[:, 2] * gamma_rr_route / timestep

    assert allclose(states_r, x_r)
    assert allclose(qout_r, expected)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("gamma", [10, 100, 10000])
@pytest.mark.parametrize("demand", [-0.1, 0, 0.1])
@pytest.mark.parametrize("timestep", [1., 0.1])
@pytest.mark.parametrize("diff", [1, 2, 3])
def test_gradbench_rainfall_runoff_tangent(siteid, gamma, demand, timestep,
                                           diff, allclose):
    x0, _, _, pin, ein = get_data(siteid)

    # run benchmarks
    gamma_rr_prod = gamma
    dgamma_rr_prod = 1. if diff == 1 else 0.
    gamma_rr_route = gamma / 5.
    dgamma_rr_route = 1. if diff == 2 else 0.
    demand_rr_route = demand
    ddemand_rr_route = 1. if diff == 3 else 0.
    out = benchmarks.gradbench_rainfall_runoff_tangent(timestep, pin, ein, x0,
                                                       gamma_rr_prod,
                                                       dgamma_rr_prod,
                                                       gamma_rr_route,
                                                       dgamma_rr_route,
                                                       demand_rr_route,
                                                       ddemand_rr_route)
    # Numerical differenciation
    eps = 5e-4
    gamma_rr_prod1 = gamma_rr_prod - dgamma_rr_prod * eps
    gamma_rr_route1 = gamma_rr_route - dgamma_rr_route * eps
    demand_rr_route1 = demand_rr_route - ddemand_rr_route * eps
    out1 = benchmarks.gradbench_rainfall_runoff(timestep, pin, ein, x0,
                                             gamma_rr_prod1,
                                             gamma_rr_route1,
                                             demand_rr_route1)

    gamma_rr_prod2 = gamma_rr_prod + dgamma_rr_prod * eps
    gamma_rr_route2 = gamma_rr_route + dgamma_rr_route * eps
    demand_rr_route2 = demand_rr_route + ddemand_rr_route * eps
    out2 = benchmarks.gradbench_rainfall_runoff(timestep, pin, ein, x0,
                                             gamma_rr_prod2,
                                             gamma_rr_route2,
                                             demand_rr_route2)
    atol = 3e-4
    for i in range(len(out1)):
        va = np.asinh(out[1 + 2 * i])
        vb = np.asinh((out2[i] - out1[i]) / 2 / eps)
        err = np.abs(va - vb)
        errmax = err.max()
        assert errmax < atol

        if vb.std() > 0:
            rho = np.corrcoef(va.squeeze(), vb.squeeze())[0, 1]
            assert rho > 1 - 1e-2

@pytest.mark.parametrize("gamma", [10, 100, 10000])
@pytest.mark.parametrize("demand", [-0.1, 0, 0.1])
@pytest.mark.parametrize("timestep", [1., 0.1])
@pytest.mark.parametrize("nreaches", [1, 2, 5])
def test_gradbench_semidist(gamma, demand, timestep, nreaches, allclose):
    nval = 1000
    pin = np.empty((nval, nreaches))
    ein = np.empty((nval, nreaches))
    areas = 100 * np.ones(nreaches)
    for isite, siteid in enumerate(data_reader.SITEIDS[:nreaches]):
        x0, _, q, p, e = get_data(siteid)
        pin[:nval, isite] = p[:nval]
        ein[:nval, isite] = e[:nval]

    # run benchmarks
    nval = len(pin)

    gammas_rr_prod = np.array([gamma] * nreaches)
    gammas_rr_route = np.array([gamma / 10] * nreaches)
    gammas_ro_route = np.array([gamma / 20] * nreaches)
    demands_rr_route = np.ones(nreaches) * demand

    out = benchmarks.gradbench_semidist(timestep, pin, ein, x0, areas,
                                       gammas_rr_prod, gammas_rr_route,
                                       gammas_ro_route, demands_rr_route)
    states_rr_p, states_rr_r, states_ro_r, qout_rr_p, qout_rr_r, qout_ro_r = out

    # Run quasoare
    amax = (1 + round(states_rr_p.max() / 0.2)) * 0.2
    alphas_rr_p, coefs_rr_p = get_coefs(2, amax)

    amax = (1 + round(states_rr_r.max() / 0.2)) * 0.2
    alphas_rr_r, coefs_rr_r = get_coefs(3, amax)

    try:
        amax = (1 + round(states_ro_r.max() / 0.2)) * 0.2
    except:
        amax = 1.
    alphas_ro_r, coefs_ro_r = get_coefs(1, amax)

    scalings = np.ones((nval, 3))
    q_states_rr_p = np.empty((nval, nreaches))
    q_states_rr_r = np.empty((nval, nreaches))
    q_states_ro_r = np.empty((nval, nreaches))
    q_qout_rr_p = np.empty((nval, nreaches))
    q_qout_rr_r = np.empty((nval, nreaches))
    q_qout_ro_r = np.empty((nval, nreaches))

    for i in range(nreaches):
        # RR Production
        scalings[:, 0] = pin[:, i] / gammas_rr_prod[i]
        scalings[:, 1] = -ein[:, i] / gammas_rr_prod[i]
        scalings[:, 2] = -1. / timestep
        _, x, fx = models.quad_model(alphas_rr_p, scalings, coefs_rr_p,
                                         x0, timestep=timestep)
        qout = pin[:, i] - (fx[:, 0] + fx[:, 2]) * gammas_rr_prod[i] / timestep
        q_states_rr_p[:, i] = x
        q_qout_rr_p[:, i] = qout

        # RR routing
        scalings[:, 0] = qout / gammas_rr_route[i]
        scalings[:, 1] = demands_rr_route[i] / gammas_rr_route[i]
        scalings[:, 2] = -1 / timestep
        _, x, fx = models.quad_model(alphas_rr_r, scalings, coefs_rr_r,
                                     x0, timestep=timestep)
        qout = qout * 0.1 - fx[:, 2] * gammas_rr_route[i] / timestep
        q_states_rr_r[:, i] = x
        q_qout_rr_r[:, i] = qout

        # Quad routing
        qin = np.zeros(nval) if i == 0 else q_qout_ro_r[:, i - 1]
        scalings[:, 0] = qin / gammas_ro_route[i]
        scalings[:, 1] = 0.
        scalings[:, 2] = -1. / timestep
        _, x, fx = models.quad_model(alphas_ro_r, scalings, coefs_ro_r,
                                     x0, timestep=timestep)

        qout = -fx[:, -1] * gammas_ro_route[i] / timestep

        q_states_ro_r[:, i] = x
        fact = areas[i] * timestep / 86.4
        q_qout_ro_r[:, i] = qout + q_qout_rr_r[:, i] * fact

    assert allclose(states_rr_p, q_states_rr_p)
    assert allclose(states_rr_r, q_states_rr_r)
    assert allclose(states_ro_r, q_states_ro_r)

    assert allclose(qout_rr_p, q_qout_rr_p)
    assert allclose(qout_rr_r, q_qout_rr_r)
    assert allclose(qout_ro_r, q_qout_ro_r)


@pytest.mark.parametrize("gamma", [10, 100, 10000])
@pytest.mark.parametrize("demand", [-0.1, 0, 0.1])
@pytest.mark.parametrize("timestep", [1., 0.1])
@pytest.mark.parametrize("nreaches", [1, 2, 5])
@pytest.mark.parametrize("diff", [1, 2, 3, 4])
def test_gradbench_semidist_tangent(gamma, demand, timestep, nreaches,
                                    diff, allclose):
    nval = 1000
    pin = np.empty((nval, nreaches))
    ein = np.empty((nval, nreaches))
    areas = 100 * np.ones(nreaches)
    for isite, siteid in enumerate(data_reader.SITEIDS[:nreaches]):
        x0, _, q, p, e = get_data(siteid)
        pin[:nval, isite] = p[:nval]
        ein[:nval, isite] = e[:nval]

    # run benchmarks
    gammas_rr_prod = np.array([gamma] * nreaches)
    zr = np.zeros(nreaches)
    o = np.array([1.] + [0.] * (nreaches - 1))
    dgammas_rr_prod = o if diff == 1 else zr
    gammas_rr_route = np.array([gamma / 10] * nreaches)
    dgammas_rr_route = o if diff == 2 else zr
    gammas_ro_route = np.array([gamma / 20] * nreaches)
    dgammas_ro_route = o if diff == 3 else zr
    demands_rr_route = np.ones(nreaches) * demand
    ddemands_rr_route = o if diff == 4 else zr

    out = benchmarks.gradbench_semidist_tangent(timestep, pin, ein, x0, areas,
                                                gammas_rr_prod,
                                                dgammas_rr_prod,
                                                gammas_rr_route,
                                                dgammas_rr_route,
                                                gammas_ro_route,
                                                dgammas_ro_route,
                                                demands_rr_route,
                                                ddemands_rr_route)

    eps = 5e-4
    gammas_rr_prod1 = gammas_rr_prod - dgammas_rr_prod * eps
    gammas_rr_route1 = gammas_rr_route - dgammas_rr_route * eps
    gammas_ro_route1 = gammas_ro_route - dgammas_ro_route * eps
    demands_rr_route1 = demands_rr_route - ddemands_rr_route * eps
    out1 = benchmarks.gradbench_semidist(timestep, pin, ein, x0, areas,
                                         gammas_rr_prod1,
                                         gammas_rr_route1,
                                         gammas_ro_route1,
                                         demands_rr_route1)

    gammas_rr_prod2 = gammas_rr_prod + dgammas_rr_prod * eps
    gammas_rr_route2 = gammas_rr_route + dgammas_rr_route * eps
    gammas_ro_route2 = gammas_ro_route + dgammas_ro_route * eps
    demands_rr_route2 = demands_rr_route + ddemands_rr_route * eps
    out2 = benchmarks.gradbench_semidist(timestep, pin, ein, x0, areas,
                                         gammas_rr_prod2,
                                         gammas_rr_route2,
                                         gammas_ro_route2,
                                         demands_rr_route2)
    atol = 1e-4
    for i in range(len(out1)):
        va = np.asinh(out[1 + 2 * i])
        vb = np.asinh((out2[i] - out1[i]) / 2 / eps)

        for j in range(va.shape[1]):
            vva, vvb = va[:, j], vb[:, j]
            err = np.abs(vva - vvb)
            errmax = err.max()
            assert errmax < atol

            if vvb.std() > 0:
                rho = np.corrcoef(vva, vvb)[0, 1]
                assert rho > 1 - 1e-2




