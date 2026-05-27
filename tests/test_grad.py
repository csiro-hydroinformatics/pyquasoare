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

from pyquasoare import integrate, models, gradients, approx
import c_pyquasoare

import data_reader

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent
LOGGER = iutils.get_logger("models", flog=FTEST / "test_grad.log")

def one(x):
    return np.ones_like(x)

def zero(x):
    return np.zeros_like(x)

RESERVOIR_FUNCTIONS = {
    "gr4jprod": [
        lambda x: 1 - x**2,
        lambda x: x * (2 - x),
        lambda x: 4**3/9**4 * x**5
        ],
    "gr4jroute": [
        lambda x: 0.9 * one(x),
        lambda x: x ** 3.5,
        lambda x: x**5 / 4
        ],
    "quadrouting": [
        lambda x: one(x),
        lambda x: zero(x),
        lambda x: x**2
        ],
    "nonlinouting": [
        lambda x: one(x),
        lambda x: zero(x),
        lambda x: x**6
        ],
    "xinanjiang": [
        lambda x: (1 - x)**0.2,
        lambda x: one(x),
        lambda x: zero(x)
        ],
    "intercept": [
        lambda x: one(x),
        lambda x: one(x),
        lambda x: zero(x)
        ]
}


@pytest.mark.parametrize("reservoir", RESERVOIR_FUNCTIONS)
def test_lin_coefficient_matrix(reservoir, allclose):
    alphas = np.linspace(0, 1, 50)
    funs = RESERVOIR_FUNCTIONS[reservoir]
    coefs = gradients.lin_coefficient_matrix(funs, alphas)

    for i, f in enumerate(funs):
        y1 = f(alphas)[:-1]
        y2 = coefs[i][:, 0] * alphas[:-1] + coefs[i][:, 1]
        assert allclose(y1, y2)



def test_lin_integrate_error():
    # Lin
    n = np.zeros(1, dtype=np.int32)
    x = np.zeros(1, dtype=np.float64)
    fx = np.zeros(6, dtype=np.float64)

    funs = [
        lambda x: np.ones_like(x),
        lambda x: x,
        lambda x: x
    ]
    alphas = np.linspace(0, 1, 50)
    coefs_lin = gradients.lin_coefficient_matrix(funs, alphas)
    timestep = 1.
    gamma = 100.
    x_start = 0.5

    qst = -np.ones(2)
    ierr = c_pyquasoare.lin_integrate(alphas, timestep, gamma,
                                      qst, coefs_lin,
                                      0, x_start,
                                      n, x, fx)
    assert ierr == 112

    qst = np.ones(2)
    coefs_lin[0] *= -1
    ierr = c_pyquasoare.lin_integrate(alphas, timestep, gamma,
                                      qst, coefs_lin,
                                      0, x_start,
                                      n, x, fx)
    assert ierr == 111


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", RESERVOIR_FUNCTIONS)
@pytest.mark.parametrize("gamma", [10, 100, 10000])
@pytest.mark.parametrize("timestep", [0.1, 1])
def test_lin_integrate(siteid, reservoir, gamma, timestep, allclose):
    # Get data
    df = data_reader.get_data(siteid, "daily").iloc[:200]
    Qstars = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
    Qstars[:, 1] *= -1 # PET = negative demand
    nval = len(Qstars)

    # Get coefs
    alphas = np.linspace(0, 1, 50)
    funs = RESERVOIR_FUNCTIONS[reservoir]
    coefs = approx.quad_coefficient_matrix(funs, alphas,
                                           approx_opt=0)
    coefs_lin = gradients.lin_coefficient_matrix(funs, alphas)

    # Loop and integrate
    x_start = 0.5
    scalings = np.zeros(3)
    for i in range(nval):
        # Quad interpol
        scalings[0] = Qstars[i, 0] / gamma
        scalings[1] = Qstars[i, 1] / gamma
        scalings[2] = - 1. / timestep
        n1, x1, fx1 = integrate.quad_integrate(alphas, scalings,
                                               coefs, 0.,
                                               x_start, timestep)
        fx1 *= gamma / timestep

        # Lin
        n2 = np.zeros(1, dtype=np.int32)
        x2 = np.zeros(1, dtype=np.float64)
        fx2 = np.zeros(6, dtype=np.float64)
        qst = np.ascontiguousarray(Qstars[i])
        ierr = c_pyquasoare.lin_integrate(alphas, timestep, gamma,
                                          qst, coefs_lin,
                                          0, x_start,
                                          n2, x2, fx2)
        assert allclose(x1, x2[0])
        assert allclose(fx1, fx2[:3])

        fxa = np.array([qst[0] - fx1[0], qst[1] - fx1[1], fx1[0] - qst[0] + fx1[2]])
        assert allclose(fxa, fx2[3:])

        x_start = x1


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", RESERVOIR_FUNCTIONS)
@pytest.mark.parametrize("gamma", [10, 100, 10000])
@pytest.mark.parametrize("timestep", [0.1, 1])
def test_lin_reservoir(siteid, reservoir, gamma, timestep, allclose):
    # Get data
    df = data_reader.get_data(siteid, "daily")
    Qstars = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
    Qstars[:, 1] *= -1 # PET = negative demand
    Qstars = np.ascontiguousarray(Qstars)
    nval = len(Qstars)

    # Get coefs
    alphas = np.linspace(0, 1, 50)
    funs = RESERVOIR_FUNCTIONS[reservoir]
    coefs = approx.quad_coefficient_matrix(funs, alphas,
                                           approx_opt=0)
    coefs_lin = np.ascontiguousarray(coefs[:, :, 1:])

    # Quadratic
    x0 = 0.5
    scalings = np.ones((nval, 3))
    scalings[:, :2] = Qstars / gamma
    scalings[:, 2] = - 1. / timestep
    n1, x1, fx1 = models.quad_model(alphas, scalings, coefs,
                                    x0, timestep)
    fx1 *= gamma / timestep

    # Lin
    n2, x2, fx2 = gradients.lin_reservoir(alphas, timestep, gamma,
                                          Qstars, coefs_lin, x0)

    assert allclose(x1, x2)

    fx1 = np.column_stack([fx1,
                           Qstars[:, 0] - fx1[:, 0],
                           Qstars[:, 1] - fx1[:, 1],
                           -Qstars[:, 0] + fx1[:, 0] + fx1[:, 2]])
    assert allclose(fx1, fx2)

    # Mass balance
    dSa = np.diff(x1) * gamma
    dSb = timestep * fx2[1:, :3].sum(axis=1)
    assert allclose(dSa, dSb)

