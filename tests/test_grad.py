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
    "gr4j_prod": [
        lambda x: 1 - x**2,
        lambda x: x * (2 - x),
        lambda x: 4**3/9**4 * x**5
        ],
    "gr4j_route": [
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


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", RESERVOIR_FUNCTIONS)
@pytest.mark.parametrize("gamma", [10, 100, 10000])
@pytest.mark.parametrize("timestep", [0.1, 1])
def test_lin_integrate(siteid, reservoir, gamma, timestep, allclose):
    # Get data
    df = data_reader.get_data(siteid, "daily")
    Qstars = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
    Qstars[:, 1] *= -1 # PET = negative demand
    nval = len(Qstars)

    # Get coefs
    alphas = np.linspace(0, 1, 50)
    funs = RESERVOIR_FUNCTIONS[reservoir]
    coefs = approx.quad_coefficient_matrix(funs, alphas,
                                           approx_opt=0)
    coefs_lin = np.ascontiguousarray(coefs[:, :, 1:])

    # Loop and integrate
    x_start = 0.5
    scalings = np.zeros(3)
    qst = np.zeros(3)
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

        x_start = x1



