from pathlib import Path
import math
import re
import pytest
import numpy as np
import pandas as pd

from itertools import product as prod

from timeit import Timer
import time

import warnings

from hydrodiy.io import csv
from pyrezeq import rezeq

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent

def get_data():
    fq = FTEST / "streamflow_423202_423206_event.csv"
    data, _ = csv.read_csv(fq, index_col=0, parse_dates=True)
    data = data.apply(lambda x: x.interpolate())
    inflows = data.FLOWUP
    outflows = data.FLOWDOWN
    return inflows, outflows


def test_piecewise_linear_approximation(allclose):
    f1 = lambda x: x**3
    f2 = lambda x: np.sin(x)
    f3 = lambda x: np.exp(-x**2)
    alphas = np.linspace(-2, 2, 50)

    for f in [f1, f2, f3]:
        coefs = rezeq.piecewise_linear_approximation(f, alphas)
        y1 = coefs[:, 0]+alphas[:-1]*coefs[:, 1]
        y2 = f(alphas[:-1])
        assert allclose(y1, y2)


def test_run_piecewise_approximation(allclose):
    f = lambda x: x**3-3*x
    nalphas = 10
    alphas = np.linspace(-2, 2, nalphas)
    coefs = rezeq.piecewise_linear_approximation(f, alphas)

    xx = np.linspace(-3, 3, 100)
    yy = rezeq.run_piecewise_approximation(xx, alphas, coefs)
    ii = np.sum(xx[:, None]-alphas[None, :]>0, axis=1)-1
    ii = np.clip(ii, 0, nalphas-2)
    expected = coefs[ii, 0]+coefs[ii, 1]*xx
    assert allclose(yy, expected)


def test_integrate_forward(allclose):
    t0 = 0
    u0 = 10.
    a = 4.
    b = -2.
    t = 1.
    u1 = rezeq.integrate_forward(t0, u0, a, b, t)
    expected = -a/b+(u0+a/b)*np.exp(b*t)
    assert allclose(u1, expected)

    u1 = rezeq.integrate_forward(t0, u0, a, b, 1e100)
    expected = -a/b
    assert allclose(u1, expected)

    b = 0.
    u1 = rezeq.integrate_forward(t0, u0, a, b, t)
    expected = u0+a*t
    assert allclose(u1, expected)


def test_integrate_inverse(allclose):
    t0 = 0
    u0 = 10.
    a = 4.
    b = -2.
    u = 4.
    t1 = rezeq.integrate_inverse(t0, u0, a, b, u)
    expected = np.log((u+a/b)/(u0+a/b))/b
    assert allclose(t1, expected)

    b = 0.
    t1 = rezeq.integrate_inverse(t0, u0, a, b, u)
    expected = (u-u0)/a
    assert allclose(t1, expected)


def test_find_alphas(allclose):
    alphas = np.linspace(0, 1, 4)
    u0 = -1.
    ialpha = rezeq.find_alpha(u0, alphas)
    assert ialpha == 0

    u0 = 1.1
    ialpha = rezeq.find_alpha(u0, alphas)
    assert ialpha == 2

    u0 = 0.2
    ialpha = rezeq.find_alpha(u0, alphas)
    assert ialpha == 0

    u0 = 0.4
    ialpha = rezeq.find_alpha(u0, alphas)
    assert ialpha == 1

    u0 = 0.7
    ialpha = rezeq.find_alpha(u0, alphas)
    assert ialpha == 2


def test_integrate_power(allclose):
    delta = 1
    nalphas = 1000
    alphas = np.linspace(0, 10, nalphas)
    lams = np.linspace(0.05, 5, 30)
    u0s = np.linspace(0.2, 10, 30)
    scalings = np.ones(1)

    # Power law
    res = []
    for lam, u0 in prod(lams, u0s):
        f = lambda x: -x**lam
        coefs = rezeq.piecewise_linear_approximation(f, alphas)
        a_matrix_noscaling = coefs[:, [0]]
        b_matrix_noscaling = coefs[:, [1]]

        u1, fluxes = rezeq.integrate(delta, u0, alphas, scalings,
                                        a_matrix_noscaling, b_matrix_noscaling)
        expected = (u0**(1-lam)+(lam-1)*delta)**(1./(1-lam))
        if not np.isnan(expected):
            err = math.asinh(expected)-math.asinh(u1)
            assert abs(err)<5e-3

            ff = (u1-u0)/delta
            assert allclose(fluxes[0], ff)


def test_integrate_linres(allclose):
    delta = 1
    nalphas = 1000
    alphas = np.linspace(0, 10, nalphas)
    taus = np.linspace(1, 100, 30)
    u0s = np.linspace(0.2, 10, 30)

    inflow = 2.5
    scalings = np.array([inflow, 1.])

    for tau, u0 in prod(taus, u0s):
        f = lambda x: -tau*x
        coefs = rezeq.piecewise_linear_approximation(f, alphas)
        a_matrix_noscaling = np.column_stack([np.ones(nalphas-1), coefs[:, [0]]])
        b_matrix_noscaling = np.column_stack([np.zeros(nalphas-1), coefs[:, [1]]])

        u1, fluxes = rezeq.integrate(delta, u0, alphas, scalings, \
                            a_matrix_noscaling, b_matrix_noscaling)

        expected = inflow/tau+(u0-inflow/tau)*math.exp(-delta*tau)
        if not np.isnan(expected):
            err = math.asinh(expected)-math.asinh(u1)
            assert abs(err)<5e-3

            ff = (u1-u0)/delta
            assert allclose(fluxes.sum(), ff)


def test_run_linres(allclose):
    delta = 1
    nalphas = 1000
    alphas = np.linspace(0, 10, nalphas)
    tau = 10.
    u0 = 1

    nval = 100
    inflows = np.exp(np.random.uniform(-4, 4, size=nval))
    inflows += inflows.mean()*2
    scalings = np.column_stack([inflows, np.ones(nval)])

    f = lambda x: -tau*x
    coefs = rezeq.piecewise_linear_approximation(f, alphas)
    a_matrix_noscaling = np.column_stack([np.ones(nalphas-1), coefs[:, [0]]])
    b_matrix_noscaling = np.column_stack([np.zeros(nalphas-1), coefs[:, [1]]])

    u1, fluxes = rezeq.run(delta, u0, alphas, scalings, \
                            a_matrix_noscaling, b_matrix_noscaling)

    assert allclose(fluxes[:, 0], inflows)

    for t in range(nval):
        expected = inflows[t]/tau+(u0-inflows[t]/tau)*math.exp(-delta*tau)
        if not np.isnan(expected):
            err = math.asinh(expected)-math.asinh(u1[t])
            assert abs(err)<5e-3

            ff = (u1[t]-u0)/delta
            assert allclose(fluxes[t].sum(), ff)

        u0 = u1[t]


def test_integrate_linquad(allclose):
    delta = 1
    nalphas = 1000
    alphas = np.linspace(0, 10, nalphas)
    thetas = np.linspace(1, 100, 30)
    u0s = np.linspace(0.2, 10, 30)
    inflow = 1.
    q0 = 1.

    for theta, u0 in prod(thetas, u0s):
        f = lambda x: -x**2
        coefs = rezeq.piecewise_linear_approximation(f, alphas)
        scalings = np.array([inflow, q0/theta**2])
        a_matrix_noscaling = np.column_stack([np.ones(nalphas-1), coefs[:, [0]]])
        b_matrix_noscaling = np.column_stack([np.zeros(nalphas-1), coefs[:, [1]]])

        u1, fluxes = rezeq.integrate(delta, u0, alphas, scalings, \
                            a_matrix_noscaling, b_matrix_noscaling)

        gamma = theta*math.sqrt(inflow/q0)
        t = math.tanh(inflow*delta/gamma)
        expected = (u0+gamma*t)/(1+u0/gamma*t)
        if not np.isnan(expected):
            err = math.asinh(expected)-math.asinh(u1)
            assert abs(err)<5e-3

            ff = (u1-u0)/delta
            assert allclose(fluxes.sum(), ff)


def test_integrate_mass_balance(allclose):
    deltas = np.linspace(0.1, 5, 10)
    nalphas = 10
    alphas = np.linspace(0, 10, nalphas)

    scalings = [1.]
    a_matrix_noscaling = [np.ones(nalphas-1)]
    b_matrix_noscaling = [np.zeros(nalphas-1)]
    f1 = lambda x: -(-1 if x<0 else 1)*math.sqrt(abs(x))
    f2 = lambda x: -x
    f3 = lambda x: -3*x**2
    for f in [f1, f2, f3]:
        coefs = rezeq.piecewise_linear_approximation(f, alphas)
        scalings.append(1.)
        a_matrix_noscaling.append(coefs[:, 0])
        b_matrix_noscaling.append(coefs[:, 1])

    u0 = 1.
    scalings = np.array(scalings)
    a_matrix_noscaling = np.column_stack(a_matrix_noscaling)
    b_matrix_noscaling = np.column_stack(b_matrix_noscaling)

    for delta in deltas:
        u1, fluxes = rezeq.integrate(delta, u0, alphas, scalings, \
                        a_matrix_noscaling, b_matrix_noscaling)
        mass_bal = u1-u0-fluxes.sum()
        assert allclose(mass_bal, 0.)


def test_integrate_non_linear_reservoir(allclose):
    delta = 1.

    alpha_min = 1313
    alpha_max = 4153
    nalphas = 3
    alphas = np.linspace(alpha_min, alpha_max, nalphas)

    theta = 3239
    q0 = 71.29
    nu = 6.
    rezfun = lambda x: -q0*(x/theta)**nu
    coefs = rezeq.piecewise_linear_approximation(rezfun, alphas)

    a_matrix_noscaling = np.column_stack([np.ones(nalphas-1), coefs[:, [0]]])
    b_matrix_noscaling = np.column_stack([np.zeros(nalphas-1), coefs[:, [1]]])
    scalings = [0.316, 1.]

    # Impact of s0 on solution
    s0 = 0.
    s1, fluxes = rezeq.integrate(delta, s0, alphas, scalings, \
                a_matrix_noscaling, b_matrix_noscaling)
    assert fluxes[1]>0

    s0 = alpha_min
    s1, fluxes = rezeq.integrate(delta, s0, alphas, scalings, \
                a_matrix_noscaling, b_matrix_noscaling)
    assert fluxes[1]<0


def test_quadrouting(allclose):
    inflows, outflows = get_data()

    delta = 1
    q0 = inflows.mean()
    theta = (inflows.cumsum()-outflows.cumsum()).max()*0.7
    s0 = 0.
    sim1 = pd.Series(rezeq.quadrouting(delta, theta, q0, s0, inflows), \
                        index=inflows.index)

    f = lambda x: -q0*(x/theta)**2
    nalphas = 500
    alpha_min = 0
    alpha_max = 2*theta
    alphas = np.linspace(alpha_min, alpha_max, nalphas)

    coefs = rezeq.piecewise_linear_approximation(f, alphas)
    a_matrix_noscaling = np.column_stack([np.ones(nalphas-1), coefs[:, [0]]])
    b_matrix_noscaling = np.column_stack([np.zeros(nalphas-1), coefs[:, [1]]])
    scalings = np.column_stack([inflows, np.ones_like(inflows)])
    s1, fluxes = rezeq.run(delta, s0, alphas, scalings, \
                            a_matrix_noscaling, b_matrix_noscaling)
    sim2 = pd.Series(-fluxes[:, 1], index=inflows.index)

    assert allclose(sim1, sim2, atol=5e-2)


def test_numrouting(allclose):
    inflows, outflows = get_data()

    delta = 1
    q0 = inflows.mean()
    theta = (inflows.cumsum()-outflows.cumsum()).max()*0.7
    s0 = 0.
    sim1 = pd.Series(rezeq.quadrouting(delta, theta, q0, s0, inflows), \
                        index=inflows.index)

    s2, f2 = rezeq.numrouting(delta, theta, q0, s0, inflows, 2.)
    sim2 = pd.Series(f2, index=inflows.index)

    assert allclose(sim1, sim2, atol=1e-8)


def test_integrate_continuity(allclose):
    delta = 1
    nalphas = 10
    alphas = np.linspace(0, 10, nalphas)
    theta = 10.
    u0 = 0.1
    inflow = 100.
    q0 = 1.

    f = lambda x: -x**2
    coefs = rezeq.piecewise_linear_approximation(f, alphas)
    scalings = np.array([inflow, q0/theta**2])
    a_matrix_noscaling = np.column_stack([np.ones(nalphas-1), coefs[:, [0]]])
    b_matrix_noscaling = np.column_stack([np.zeros(nalphas-1), coefs[:, [1]]])
    b_matrix_noscaling[:, 1] += np.random.uniform(0.2, 1, nalphas-1)

    with pytest.raises(ValueError, match="integrate returns"):
        u1, fluxes = rezeq.integrate(delta, u0, alphas, scalings, \
                        a_matrix_noscaling, b_matrix_noscaling)
