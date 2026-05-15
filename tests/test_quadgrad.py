from pathlib import Path
from itertools import product as prod
import time
import math
import re
import pytest

import numpy as np
import pandas as pd

from hydrodiy.io import iutils

from pyquasoare import approx, models, quadgrad, steady

import data_reader

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent

def get_data(siteid):
    df = data_reader.get_data(siteid, "daily")
    df = df.interpolate()
    qin = df.iloc[:, -1].values
    x0 = 0.5
    timestep = 1.
    return x0, timestep, qin


def get_scalings(qin, gamma):
    nval = len(qin)
    scalings = np.ones((nval, 3))
    scalings[:, 0] = qin / gamma
    return scalings


def get_coefs(amax):
    fluxes = [
        lambda x: np.ones_like(x),
        lambda x: np.zeros_like(x),
        lambda x: -x**2
        ]
    alphas = np.linspace(0, amax, 3)
    return alphas, approx.quad_coefficient_matrix(fluxes, alphas)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("gamma", [1, 100, 10000])
def test_quadgrad_single_reach(siteid, gamma, allclose):
    x0, timestep, qin = get_data(siteid)
    scalings = get_scalings(qin, gamma)

    # run quadgrad
    gammas = np.array([gamma])
    states, qout = quadgrad.quadgrad(timestep, qin, x0, gammas)
    qout = qout.squeeze()
    states = states.squeeze()

    # Run quasoare
    amax = (1 + round(states.max() / 0.2)) * 0.2
    alphas, coefs = get_coefs(amax)
    _, x1, fx = models.quad_model(alphas, scalings, coefs, x0,
                                  timestep=timestep)
    expected = -fx[:, -1] * gamma

    assert allclose(states, x1)
    assert allclose(qout, expected)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("gamma", [1, 100, 10000])
def test_quadgrad_multi_reach(siteid, gamma, allclose):
    x0, timestep, qin = get_data(siteid)
    nreaches = 5

    # run quadgrad
    gammas = np.array([gamma] * nreaches)
    states, qout = quadgrad.quadgrad(timestep, qin, x0, gammas)

    # Run quasoare
    amax = (1 + round(states.max() / 0.2)) * 0.2
    alphas, coefs = get_coefs(amax)

    sexpected = np.zeros_like(states)
    qexpected = np.zeros_like(qout)
    for i in range(nreaches):
        qup = qin if i == 0 else qexpected[:, i - 1]
        scalings = get_scalings(qup, gamma)
        _, x1, fx = models.quad_model(alphas, scalings, coefs, x0,
                                      timestep=timestep)
        qexpected[:, i] = -fx[:, -1] * gamma
        sexpected[:, i] = x1

    assert allclose(states, sexpected)
    assert allclose(qout, qexpected)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("gamma", [1, 100, 10000])
def test_quadgrad_tangent(siteid, gamma, allclose):
    x0, timestep, qin = get_data(siteid)
    eps = 1e-6

    # run quadgrad
    gammas = np.array([gamma - eps])
    states_1, qout_1 = quadgrad.quadgrad(timestep, qin, x0, gammas)
    qout_1 = qout_1.squeeze()
    states_1 = states_1.squeeze()

    gammas = np.array([gamma + eps])
    states_2, qout_2 = quadgrad.quadgrad(timestep, qin, x0, gammas)
    qout_2 = qout_2.squeeze()
    states_2 = states_2.squeeze()

    sexpected = (states_2 - states_1) / 2 / eps
    qexpected = (qout_2 - qout_1) / 2 / eps

    # run quadgrad tangent
    gammas = np.array([gamma])
    gammas_diff = np.array([1.])
    states, qout, states_diff, qout_diff = quadgrad.quadgrad_tangent(timestep, qin, x0,
                                                                     gammas,
                                                                     gammas_diff)
    assert allclose(sexpected, states_diff.squeeze())
    assert allclose(qexpected, qout_diff.squeeze(), atol=5e-7)
