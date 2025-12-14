from pathlib import Path
from itertools import product as prod
import math
import re
import pytest
import numpy as np
import pandas as pd
import time

from hydrodiy.io import iutils

from pyquasoare import approx, slow
import c_pyquasoare

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent

NCASES = 13

LOGGER = iutils.get_logger("approx", flog=FTEST / "test_approx.log")

# ----- FIXTURES ---------------------------------------------------
@pytest.fixture(scope="module", \
            params=["x2", "x4", "x6", "x8", "tanh", "exp", "sin", \
                            "recip", "recipquad", "runge", "stiff", \
                            "ratio", "logistic", "genlogistic"])
def reservoir_function(request, selfun):
    name = request.param
    if name!=selfun and selfun!="":
        pytest.skip("Function skipped")

    # Alpha interval
    alpha0, alpha1 = (0., 1.2)

    # sol is the analytical solution of ds/dt = inflow+fun(s)
    if name == "x2":
        fun = lambda x: np.clip(-x**2, -np.inf, 0)
        dfun = lambda x: np.clip(-2*x, -np.inf, 0)
        inflow = 1.
        sol = lambda t, s0: (s0+np.tanh(t))/(1+s0*np.tanh(t))

    elif name == "x4":
        fun = lambda x: np.clip(-x**4, -np.inf, 0)
        dfun = lambda x: np.clip(-4*x**3, -np.inf, 0)
        inflow = 0.
        sol = lambda t, s0: s0/(1+3*t*s0**3)**(1./3)

    elif name == "x6":
        fun = lambda x: np.clip(-x**6, -np.inf, 0)
        dfun = lambda x: np.clip(-6*x**5, -np.inf, 0)
        inflow = 0.
        sol = lambda t, s0: (1./s0**5+5*t)**(-1./5)

    elif name == "x8":
        fun = lambda x: np.clip(-x**8, -np.inf, 0)
        dfun = lambda x: np.clip(-8*x**6, -np.inf, 0)
        inflow = 0.
        sol = lambda t, s0: (1./s0**7+7*t)**(-1./7)

    elif name == "tanh":
        alpha0, alpha1 = (-1.25, 0.75)
        a, b = 0.5, 10
        fun = lambda x: -np.tanh(a+b*x)
        dfun = lambda x: b*(np.tanh(a+b*x)**2-1)
        inflow = 0.
        sol = lambda t, s0: (np.arcsinh(np.exp(-t*b)*np.sinh(a+b*s0))-a)/b

    elif name == "exp":
        fun = lambda x: -np.exp(x)
        dfun = lambda x: -np.exp(x)
        inflow = 1.
        sol = lambda t, s0: s0+t-np.log(1-(1-np.exp(t))*math.exp(s0))

    elif name == "stiff":
        lam = 100
        fun = lambda x: -lam*x
        dfun = lambda x: -lam
        inflow = 1.
        sol = lambda t, s0: 1./lam-(1./lam-s0)*np.exp(-lam*t)

    elif name == "sin":
        alpha0, alpha1 = (0.0, 1.0)
        w = 2*math.pi
        fun = lambda x: np.sin(w*x)
        dfun = lambda x: -w*np.cos(w*x)
        inflow = 0.
        tmp = lambda t, s0: np.arccos((np.cos(w*s0)-np.tanh(w*t)) \
                                        /(1-np.cos(w*s0)*np.tanh(w*t)))/w
        sol = lambda t, s0: tmp(t, s0) if math.sin(w*s0)>0 else 2*math.pi/w-tmp(t, s0)

    elif name == "recip":
        alpha0, alpha1 = (0., 1.0)
        offset = 0.05
        fun = lambda x: -offset/(1+offset-x)
        dfun = lambda x: offset/(1+offset-x)**2
        inflow = 0.
        sol = None

    elif name == "recipquad":
        alpha0, alpha1 = (0., 1.0)
        offset = 0.05
        fun = lambda x: -offset**2/(1+offset-x)**2
        dfun = lambda x: 2*offset**2/(1+offset-x)**3
        inflow = 0.
        sol = None

    elif name == "runge":
        alpha0, alpha1 = (-1., 3.)
        fun = lambda x: 1./(1+x**2)
        dfun = lambda x: -2*x/(1+x**2)**2
        inflow = 0.
        # solution of s1^3+3s1 = 3t+s0^3+3s0
        Q = lambda t, s0: -3*t-3*s0-s0**3
        sol = lambda t, s0: np.cbrt(-Q(t,s0)/2+np.sqrt(Q(t,s0)**2/4+1))\
                            +np.cbrt(-Q(t,s0)/2-np.sqrt(Q(t,s0)**2/4+1))

    elif name == "ratio":
        n = 3
        fun = lambda x: 1./x**(n-1)-x
        dfun = lambda x: (1-n)/x**n-1
        inflow = 0.
        sol = lambda t, s0: (1-np.exp(-n*t)*(1-s0**n))**(1./n)
        alpha0, alpha1 = 5e-2, 1.

    elif name == "logistic":
        lam = 0.1
        fun = lambda x: lam*x*(1-x)
        dfun = lambda x: lam*(1-2*x)
        inflow = 0.
        sol = lambda t, s0: s0*np.exp(lam*t)/(1-s0+s0*np.exp(lam*t))
        alpha0, alpha1 = 0., 3.

    elif name == "genlogistic":
        #K, nu, alpha = 10., 0.8, 2
        K, nu, alpha = 10., 3., 1.
        fun = lambda x: alpha*(1-(x/K)**nu)*x
        dfun = lambda x: alpha*(1-(x/K)**nu)-alpha*nu*(x/K)**nu
        inflow = 0.
        sol = lambda t, s0: K/(1+((K/s0)**nu-1)*np.exp(-alpha*nu*t))**(1./nu)
        alpha0, alpha1 = 0., K

    return name, fun, dfun, sol, inflow, (alpha0, alpha1)


@pytest.fixture(scope="module", params=list(range(1, NCASES+1)))
def generate_samples(ntry, selcase, request):
    case = request.param
    assert case in list(range(1, NCASES+1))
    if selcase>0 and case!=selcase:
        pytest.skip("Configuration skipped")

    Tmax = 20

    # Parameters
    v0, v1 = -5*np.ones(3), 5*np.ones(3)

    if case == 1:
        name = "a and b are zero"
        v0[:-1] = 0
        v1[:-1] = 0
    elif case == 2:
        name = "a is zero"
        v0[0] = 0
        v1[0] = 0

    rnd = np.random.uniform(0, 1, size=(ntry, 3))
    params = v0[None, :]+(v1-v0)[None, :]*rnd

    # Refine cases
    if case == 3:
        name = "Determinant is null"
        params[:, 2] = params[:, 1]**2/4/params[:, 0]

    elif case == 4:
        name = "Determinant is negative"
        a, b, c = params.T
        Delta = -np.abs(b**2-4*a*c)
        qD = np.sqrt(-Delta)/2
        c = (b**2-Delta)/4./a
        params[:, 2] = c

    elif case in [5, 12]:
        name = "Determinant is positive" if case==5 \
                        else "edge case for atanh"
        a, b, c = params.T
        Delta = np.abs(b**2-4*a*c)
        c = (b**2-Delta)/4./a
        params[:, 2] = c

        if case == 12:
            ssr = b/2./a
            qD = np.sqrt(Delta)/2.
            s0s = -ssr+np.abs(qD/a)*np.random.uniform(0., 1., len(qD))

    elif case == 6:
        name = "General"

    elif case == 7:
        name = "General+large scale"
        params *= 1000

    elif case == 8:
        name = "General+low scale"
        params /= 1000

    elif case == 9:
        s0s = np.random.uniform(-5, 5, ntry)
        name = "a and b close to zero"
        eps = 1e-9
        params[:, :2] = np.random.uniform(eps, 5*eps, size=(len(params), 2))

    elif case == 10:
        s0s = np.random.uniform(-5, 5, ntry)
        name = "a close to zero"
        eps = 1e-9
        params[:, 0] = np.random.uniform(eps, 5*eps, size=len(params))

    elif case == 11:
        name = "equality s0 = -b/2a"
        a, b, c = params.T
        s0s = -b/2./a

    elif case == 13:
        name = "a zero and b close to zero"
        params[:, 0] = 0
        eps = 1e-9
        params[:, 1] = np.random.uniform(eps, 5*eps, size=len(params))


    if not case in [9, 10, 11, 12]:
        s0s = np.random.uniform(-5, 5, ntry)
        a, b, c = params.T
        idx = np.abs(a)>0
        if idx.sum()>0:
            Delta = np.abs(b**2-4*a*c)
            qD = np.sqrt(Delta)/2
            qD[np.abs(Delta)==0] = 3.
            ssr = b/2/a
            s0s[idx] = -ssr+qD/a*np.random.uniform(-2, 2, idx.sum())

    return f"{name:20}", case, params, s0s, Tmax


# ----- TESTS --------------------------------------------

def test_accuracy():
    eps = abs(approx.QUASOARE_ACCURACY)
    assert eps>0

def test_get_nan():
    nan = c_pyquasoare.get_nan()
    assert np.isnan(nan)

def test_get_inf():
    inf = c_pyquasoare.get_inf()
    assert inf>0
    assert np.isinf(inf)

def test_function_runtime():
    pytest.skip("no useful")
    nrepeat = 5000000
    LOGGER.info("")
    funs = ["x2", "x4", "x6", "log", "exp", "tan", "tanh"]
    runtimes = np.zeros(len(funs))
    for ifun, fun in enumerate(funs):
        # Argument range
        if fun == "log":
            x0, x1 = 1e-20, 20
        elif fun == "tan":
            x0, x1 = -math.pi/2+1e-5, math.pi/2-1e-5
        else:
            x0, x1 = -20, 20

        x, dx = x0, (x1-x0)/(nrepeat-1)

        # Repeated execution
        start = time.time()
        for i in range(nrepeat):
            if fun == "x2":
                y = x*x
            elif fun == "x4":
                y = x*x*x*x
            elif fun == "x6":
                y = x*x*x*x*x*x
            elif fun == "log":
                y = math.log(x)
            elif fun == "exp":
                y = math.exp(x)
            elif fun == "tan":
                y = math.tan(x)
            elif fun == "tanh":
                y = math.tanh(x)
            x += dx

        rt = (time.time()-start)*1e3
        runtimes[ifun] = rt

    for ifun, fun in enumerate(funs):
        rt = runtimes[ifun]
        rx2 = rt/runtimes[0]*100
        mess = f"Runtime of fun {fun:4s} : {rt:2.2e}ms ({rx2:0.0f}% of x2)"
        LOGGER.info(mess)


def test_quad_fun(allclose, generate_samples):
    cname, case, params, s0s, Tmax = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("Skip param config")

    for itry, (param, s0) in enumerate(zip(params, s0s)):
        o = approx.quad_fun(param, s0)
        a, b, c = param
        expected = a*s0**2+b*s0+c
        assert allclose(o, expected)

        o = approx.quad_grad(param, s0)
        expected = 2*a*s0+b
        assert allclose(o, expected)


def test_quad_coefficients(allclose, reservoir_function):
    fname, fun, dfun, _, _, (alpha0, alpha1) = reservoir_function

    alphas = np.array([alpha0, alpha1]).astype(float)
    f0 = float(fun(alpha0))
    f1 = float(fun(alpha1))
    falphas = np.array([f0, f1])
    xm = (alpha0+alpha1)/2
    fm = fun(xm)
    fmid = np.array([fm])

    a0, b0, c0 = approx.quad_coefficients(alphas, falphas, fmid, 0)[0]
    a1, b1, c1 = approx.quad_coefficients(alphas, falphas, fmid, 1)[0]
    a2, b2, c2 = approx.quad_coefficients(alphas, falphas, fmid, 2)[0]

    a, b, c = approx.quad_coefficients(alphas, falphas, fmid)[0]
    assert allclose([a, b, c], [a1, b1, c1])

    # Linear function
    assert allclose(a0, 0)

    # Check function on bounds
    for x in [alpha0, alpha1]:
        ft = fun(x)
        for param in zip([a0, a1, a2], [b0, b1, b2], \
                                    [c0, c1, c2]):
            p = np.array(param).astype(float)
            xx = np.array([x]).astype(float)
            fa = approx.quad_fun(p, xx)
            assert allclose(ft, fa)

    v0, v1 = (f0+3*f1)/4, (f1+3*f0)/4
    v0, v1 = min(v0, v1), max(v0, v1)
    expected = max(v0, min(v1, fm))
    fa = approx.quad_fun(np.array([a1, b1, c1]), xm)
    assert allclose(fa, expected, atol=1e-10)

    # Linear function goes through mid point
    fa = approx.quad_fun(np.array([a0, b0, c0]), xm)
    expected = (f1+f0)/2
    assert allclose(fa, expected, atol=1e-10)

    # Vectorisation
    alphas = np.linspace(alpha0, alpha1, 10)
    falphas = np.array([fun(a) for a in alphas])
    fmid = np.array([fun((alphas[i] + alphas[i+1]) / 2) for i in range(9)])
    coefs = approx.quad_coefficients(alphas, falphas, fmid)


def test_quad_coefficients_edge_cases(allclose):
    # Identical band limits
    alphas = np.array([1., 1.])
    falphas = np.array([1., 1.])
    fmid = np.array([1.])
    with pytest.raises(ValueError, match="not increasing"):
        approx.quad_coefficients(alphas, falphas, fmid)

    # high value within interval [f0, f1] -> monotonous
    alphas = np.array([0., 1.])
    falphas = np.array([1., 3.])
    fmid = np.array([2.9])
    co = approx.quad_coefficients(alphas, falphas, fmid)[0]
    fm2 = approx.quad_fun(co, 0.5)
    assert allclose(fm2, (falphas[0] + 3 * falphas[1]) / 4.)

    # high value outside interval [f0, f1] -> non mononotous
    fmid = np.array([3.5])
    co = approx.quad_coefficients(alphas, falphas, fmid, 2)[0]
    fm2 = approx.quad_fun(co, 0.5)
    assert allclose(fm2, fmid[0])

    # low value within interval -> monotonous
    falphas = np.array([1., -10.])
    fmid = np.array([-9.])
    co = approx.quad_coefficients(alphas, falphas, fmid)[0]
    fm2 = approx.quad_fun(co, 0.5)
    assert allclose(fm2, (falphas[0] + 3 * falphas[1]) / 4.)

    # low value outside interval -> non monotonous
    falphas = np.array([1., -10.])
    fmid = np.array([-11.])
    co = approx.quad_coefficients(alphas, falphas, fmid, 2)[0]
    fm2 = approx.quad_fun(co, 0.5)
    assert allclose(fm2, fmid[0])


def test_quad_coefficient_matrix(allclose, reservoir_function):
    fname, fun, dfun, sol, inflow, (alpha0, alpha1) = reservoir_function
    funs = [lambda x: inflow * np.ones_like(x), fun]

    nalphas = 21
    alphas = np.linspace(alpha0, alpha1, nalphas)

    da = 0.005 if fname in ["recip", "recipquad", "ratio"] else (alpha1-alpha0)/10
    s = np.linspace(alpha0-da, alpha1+da, 1000)
    ftrue = fun(s)
    isin = (s >= alpha0) & (s <= alpha1)

    # Check max nfluxes
    nf = approx.QUASOARE_NFLUXES_MAX
    fs = [lambda x: x**a for a in np.linspace(1, 2, nf+1)]
    with pytest.raises(ValueError, match="nfluxes"):
        coefs = approx.quad_coefficient_matrix(fs, alphas)

    coefs = approx.quad_coefficient_matrix(funs, alphas)
    coefs_copy = coefs.copy()

    # Check extrapolations
    out = approx.quad_fun_from_matrix(alphas, coefs, s)
    assert allclose(coefs, coefs_copy)
    fapprox = out[:, 1]

    ilow = s<alpha0
    y0 = approx.quad_fun(coefs[1, 0], alpha0)
    dy0 = approx.quad_grad(coefs[1, 0], alpha0)
    expected = y0 + dy0 * (s - alpha0)
    assert allclose(fapprox[ilow], expected[ilow])

    ihigh = s>alpha1
    y1 = approx.quad_fun(coefs[1, -1], alpha1)
    dy1 = approx.quad_grad(coefs[1, -1], alpha1)
    expected = y1 + dy1 * (s - alpha1)
    assert allclose(fapprox[ihigh], expected[ihigh])

    # Check interpolation
    out = approx.quad_fun_from_matrix(alphas, coefs, alphas)
    assert allclose(out[:, 1], fun(alphas))


def test_quad_vs_lin(allclose, reservoir_function):
    fname, fun, dfun, sol, inflow, (alpha0, alpha1) = reservoir_function
    if fname in ["x2", "stiff", "logistic"]:
        # Skip x2, stiff and logistic which are perfect match with quadratic functions
        pytest.skip("Skip function")

    funs = [lambda x: inflow + fun(x)]
    nalphas = 11
    alphas = np.linspace(alpha0, alpha1, nalphas)

    coefs = approx.quad_coefficient_matrix(funs, alphas)
    s = np.linspace(alpha0, alpha1, 10000)
    out = approx.quad_fun_from_matrix(alphas, coefs, s)
    fapprox = out[:, 0]

    coefs = approx.quad_coefficient_matrix(funs, alphas, 0)
    out = approx.quad_fun_from_matrix(alphas, coefs, s)
    fapprox_lin = out[:, 0]

    ftrue = funs[0](s)
    rmse = math.sqrt(((fapprox-ftrue)**2).mean())
    rmse_lin = math.sqrt(((fapprox_lin-ftrue)**2).mean())

    # We want to make sure quad is always worse than app
    ratio_thresh = {
        "x4": 2e-2, \
        "x6": 5e-2, \
        "x8": 1e-1, \
        "tanh": 5e-1, \
        "exp": 1e-2, \
        "sin": 8e-1, \
        "recip": 3e-1, \
        "recipquad": 5e-1, \
        "runge": 9e-1, \
        "ratio": 5e-1, \
        "genlogistic": 2e-1
    }
    ratio = rmse / rmse_lin
    assert ratio < ratio_thresh[fname]
    LOGGER.info("")
    LOGGER.info(f"[{fname}] quad vs lin: error ratio={ratio:2.2e}")

