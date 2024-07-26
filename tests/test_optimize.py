from pathlib import Path
import math
import re
import pytest
import numpy as np
import pandas as pd
from scipy.special import expit
import scipy.integrate as integrate

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from itertools import product as prod

import time

import warnings

from hydrodiy.io import csv
from pyrezeq import rezeq, optimize

from test_rezeq import reservoir_function

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent


def test_optimize_nu_and_epsilon(allclose, reservoir_function):
    fname, fun, dfun, sol, inflow, (alpha0, alpha1) = reservoir_function
    funs = [lambda x: inflow+fun(x)]
    nalphas = 21
    alphas = np.linspace(alpha0, alpha1, nalphas)
    nu, epsilon, amat, bmat, cmat = optimize.optimize_nu_and_epsilon(funs, alphas)

    s = np.linspace(alpha0, alpha1, 10000)
    out = rezeq.approx_fun_from_matrix(alphas, nu, amat, bmat, cmat, s)
    fapprox = out[:, 0]
    ftrue = funs[0](s)
    rmse = math.sqrt(((fapprox-ftrue)**2).mean())

    rmse_thresh = {
        "x2": 1e-9, \
        "x4": 1e-5, \
        "x6": 1e-4, \
        "x8": 1e-4, \
        "tanh": 1e-2, \
        "exp": 1e-7,
        "sin": 1e-3, \
        "recip": 1e-2, \
        "recipquad": 1e-1, \
        "runge": 1e-4, \
        "stiff": 1e-7, \
        "ratio": 2
    }
    assert rmse<rmse_thresh[fname]
    print(f"\ncoef matrix - fun({alpha0:0.2f},{alpha1:0.2f})={fname}: "\
                    +f"rmse={rmse:3.3e} (nu={nu:0.2f}, eps={epsilon:0.2f})")


def test_optimize_vs_quad(allclose, reservoir_function):
    fname, fun, dfun, sol, inflow, (alpha0, alpha1) = reservoir_function
    if fname in ["x2", "stiff"]:
        # Skip x2 and stiff which are perfect match with quadratic functions
        pytest.skip("Skip function")

    funs = [lambda x: inflow+fun(x)]
    nalphas = 21
    alphas = np.linspace(alpha0, alpha1, nalphas)
    nu, epsilon, amat, bmat, cmat = optimize.optimize_nu_and_epsilon(funs, alphas)
    s = np.linspace(alpha0, alpha1, 10000)
    out = rezeq.approx_fun_from_matrix(alphas, nu, amat, bmat, cmat, s)
    fapprox = out[:, 0]
    ftrue = funs[0](s)
    rmse = math.sqrt(((fapprox-ftrue)**2).mean())

    # quadratic interpolation
    quad = lambda x, p: p[0]+p[1]*x+p[2]*x**2
    fquad = np.nan*np.zeros_like(s)
    for j in range(nalphas-1):
        a0, a1 = alphas[[j, j+1]]
        aa = [a0, (a0+a1)/2, a1]
        X = np.array([[quad(a, [1, 0, 0]), quad(a, [0, 1, 0]), \
                            quad(a, [0, 0, 1])] for a in aa])
        Y = np.array([fun(a) for a in aa])
        pq = np.linalg.solve(X, Y)
        idx = (s>=a0-1e-10)&(s<=a1+1e-10)
        fquad[idx] = quad(s[idx], pq)

    rmse_quad = math.sqrt(((fquad-ftrue)**2).mean())

    # We want to make sure quad is always worse than app
    ratio = rmse/rmse_quad

    ratio_thresh = {
        "x4": 0.4, \
        "x6": 0.3, \
        "x8": 0.2, \
        "tanh": 1.0+1e-4, \
        "exp": 1e-10, \
        "sin": 1.0+1e-4, \
        "recip": 0.95, \
        "recipquad": 0.96, \
        "runge": 1.0+1e-4, \
        "ratio": 0.96
    }
    assert ratio<ratio_thresh[fname]
    print("")
    print(f"approx vs quad error ratio for fun={fname}: {ratio:2.2e}")


