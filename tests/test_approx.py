from pathlib import Path
from itertools import product as prod
import math
import re
import pytest
import numpy as np
import pandas as pd

from hydrodiy.io import iutils

from pyrezeq import approx, slow
import c_pyrezeq

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent

NCASES = 15

LOGGER = iutils.get_logger("approx", flog=FTEST / "test_approx.log")

# ----- FIXTURES ---------------------------------------------------
@pytest.fixture(scope="module", \
            params=["x2", "x4", "x6", "x8", "tanh", "exp", "sin", \
                            "recip", "recipquad", "runge", "stiff", "ratio"])
def reservoir_function(request, selfun):
    name = request.param
    if name!=selfun and selfun!="":
        pytest.skip("Function skipped")

    # Alpha interval
    alpha0, alpha1 = (0., 1.2)

    # sol is the analytical solution of ds/dt = inflow+fun(s)
    if name == "x2":
        sol = lambda t, s0: (s0+np.tanh(t))/(1+s0*np.tanh(t))
        return name, lambda x: -x**2, lambda x: -2*x, sol, 1.,\
                    (alpha0, alpha1)

    elif name == "x4":
        sol = lambda t, s0: s0/(1+3*t*s0**3)**(1./3)
        return name, lambda x: -x**4, lambda x: -4*x**3, sol, 0., \
                    (alpha0, alpha1)

    elif name == "x6":
        sol = lambda t, s0: (1./s0**5+5*t)**(-1./5)
        return name, lambda x: -x**6, lambda x: -6*x**5, sol, 0., \
                    (alpha0, alpha1)

    elif name == "x8":
        sol = lambda t, s0: (1./s0**7+7*t)**(-1./7)
        return name, lambda x: -x**8, lambda x: -8*x**6, sol, 0., \
                    (alpha0, alpha1)

    elif name == "tanh":
        a, b = 0.5, 10
        sol = lambda t, s0: (np.arcsinh(np.exp(-t*b)*np.sinh(a+b*s0))-a)/b
        alpha0, alpha1 = (-1., 1.)
        return name, lambda x: -np.tanh(a+b*x), \
                            lambda x: b*(np.tanh(a+b*x)**2-1), sol, 0., \
                            (alpha0, alpha1)

    elif name == "exp":
        sol = lambda t, s0: s0+t-np.log(1-(1-np.exp(t))*math.exp(s0))
        return name, lambda x: -np.exp(x), lambda x: -np.exp(x), sol, 1., \
                    (alpha0, alpha1)

    elif name == "stiff":
        lam = 100
        sol = lambda t, s0: 1./lam-(1./lam-s0)*np.exp(-lam*t)
        return name, lambda x: -lam*x, lambda x: -lam, sol, 1., \
                    (alpha0, alpha1)

    elif name == "sin":
        alpha0, alpha1 = (0.0, 1.0)
        w = 2*math.pi
        tmp = lambda t, s0: np.arccos((np.cos(w*s0)-np.tanh(w*t)) \
                                        /(1-np.cos(w*s0)*np.tanh(w*t)))/w
        sol = lambda t, s0: tmp(t, s0) if math.sin(w*s0)>0 else 2*math.pi/w-tmp(t, s0)
        return name, lambda x: np.sin(w*x), lambda x: -w*np.cos(w*x), sol, 0.,\
                    (alpha0, alpha1)

    elif name == "recip":
        alpha0, alpha1 = (0., 1.0)
        offset = 0.05
        return name, lambda x: -offset/(1+offset-x), lambda x: offset/(1+offset-x)**2, \
                            None, 0., (alpha0, alpha1)

    elif name == "recipquad":
        alpha0, alpha1 = (0., 1.0)
        offset = 0.05
        return name, lambda x: -offset**2/(1+offset-x)**2, lambda x: 2*offset**2/(1+offset-x)**3, \
                            None, 0., (alpha0, alpha1)

    elif name == "runge":
        alpha0, alpha1 = (-1, 3)
        # solution of s1^3+3s1 = 3t+s0^3+3s0
        Q = lambda t, s0: -3*t-3*s0-s0**3
        sol = lambda t, s0: np.cbrt(-Q(t,s0)/2+np.sqrt(Q(t,s0)**2/4+1))\
                            +np.cbrt(-Q(t,s0)/2-np.sqrt(Q(t,s0)**2/4+1))
        return name, lambda x: 1./(1+x**2), lambda x: -2*x/(1+x**2)**2, \
                        sol, 0., (alpha0, alpha1)

    elif name == "ratio":
        n = 3
        sol = lambda t, s0: (1-np.exp(-n*t)*(1-s0**n))**(1./n)
        alpha0, alpha1 = 5e-2, 1.
        return name, lambda x: 1./x**(n-1)-x, lambda x: (1-n)/x**n-1, sol, 0., \
                    (alpha0, alpha1)


@pytest.fixture(scope="module", params=list(range(1, NCASES+1)))
def generate_samples(ntry, selcase, request):
    case = request.param
    assert case in list(range(1, NCASES+1))
    if selcase>0 and case!=selcase:
        pytest.skip("Configuration skipped")

    v0, v1 = -5*np.ones(3), 5*np.ones(3)

    if case == 1:
        name = "all zero except a"
        v0[1:] = 0
        v1[1:] = 0
    if case == 2:
        name = "all zero except c"
        v0[:2] = 0
        v1[:2] = 0
    if case == 3:
        name = "all zero except b"
        v0[[0, 2]] = 0
        v1[[0, 2]] = 0
    elif case == 4:
        name = "a is zero"
        v0[0] = 0
        v1[0] = 0
    elif case == 5:
        name = "b is zero"
        v0[1] = 0
        v1[1] = 0
    elif case == 6:
        name = "c is zero"
        v0[2] = 0
        v1[2] = 0

    rnd = np.random.uniform(0, 1, size=(ntry, 3))
    params = v0[None, :]+(v1-v0)[None, :]*rnd

    eps = approx.REZEQ_EPS
    if case == 7:
        name = "Determinant is null"
        params[:, 2] = params[:, 0]**2/4/params[:, 1]
    elif case == 8:
        name = "linear constraint b=-c"
        params[:, 2] = -params[:, 1]
    elif case == 9:
        name = "linear constraint b=c"
        params[:, 2] = params[:, 1]
    elif case ==10:
        name = "General case"
    elif case ==11:
        name = "General case with large scaling"
        params *= 1000
    elif case ==12:
        name = "General case with low scaling"
        params /= 1000
    elif case ==13:
        name = "a close to zero"
        params[:, 0] = np.random.uniform(2*eps, 3*eps, size=len(params))
    elif case ==14:
        name = "b close to zero"
        params[:, 1] = np.random.uniform(2*eps, 3*eps, size=len(params))
    elif case ==15:
        name = "c close to zero"
        params[:, 2] = np.random.uniform(2*eps, 3*eps, size=len(params))

    # Other data
    nus = np.exp(np.random.uniform(-2, 2, size=ntry))
    s0s = np.random.uniform(-5, 5, size=ntry)
    Tmax = 20

    return name, case, params, nus, s0s, Tmax



# ----- TESTS --------------------------------------------

def test_get_nan():
    nan = c_pyrezeq.get_nan()
    assert np.isnan(nan)


def test_get_inf():
    inf = c_pyrezeq.get_inf()
    assert inf>0
    assert np.isinf(inf)


def test_approx_fun(allclose, generate_samples):
    cname, case, params, nus, s0s, Tmax = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("Skip param config")

    for itry, ((a, b, c), nu, s0) in enumerate(zip(params, nus, s0s)):
        ds = approx.approx_fun(nu, a, b, c, s0s)
        expected = a+b*np.exp(-nu*s0s)+c*np.exp(nu*s0s)
        assert allclose(ds, expected)

        ds_slow = [slow.approx_fun(nu, a, b, c, s) for s in s0s]
        assert allclose(ds_slow, expected)

        jac = approx.approx_jac(nu, a, b, c, s0s)
        expected = -nu*b*np.exp(-nu*s0s)+nu*c*np.exp(nu*s0s)
        assert allclose(jac, expected)

        jac_slow = [slow.approx_jac(nu, a, b, c, s) for s in s0s]
        assert allclose(jac_slow, expected)



def test_get_coefficients(allclose, reservoir_function):
    fname, fun, dfun, _, _, (alpha0, alpha1) = reservoir_function
    nus = [0.01, 0.1, 1, 5]

    for nu in nus:
        a, b, c, corrected = approx.get_coefficients(fun, alpha0, alpha1, nu)

        # Check function on bounds
        assert allclose(approx.approx_fun(nu, a, b, c, alpha0), fun(alpha0), atol=1e-7)
        assert allclose(approx.approx_fun(nu, a, b, c, alpha1), fun(alpha1), atol=1e-7)

        # Check mid point values
        if not corrected:
            eps = 0.5
            x = (1-eps)*alpha0+eps*alpha1
            assert allclose(approx.approx_fun(nu, a, b, c, x), fun(x))


def test_get_coefficients_matrix(allclose, reservoir_function):
    fname, fun, dfun, sol, inflow, (alpha0, alpha1) = reservoir_function
    funs = [lambda x: inflow, fun]

    nalphas = 21
    alphas = np.linspace(alpha0, alpha1, nalphas)
    nu = 1
    epsilon = 0.5

    da = 0.005 if fname in ["recip", "recipquad", "ratio"] else (alpha1-alpha0)/10
    s = np.linspace(alpha0-da, alpha1+da, 1000)
    ftrue = fun(s)
    isin = (s>=alpha0)&(s<=alpha1)

    # Check max nfluxes
    n, eps = np.ones(nalphas-1), 0.5
    nf = approx.REZEQ_NFLUXES_MAX
    fs = [lambda x: x**a for a in np.linspace(1, 2, nf+1)]
    with pytest.raises(ValueError, match="nfluxes"):
        amat, bmat, cmat = approx.get_coefficients_matrix(\
                                                        fs, alphas, n)

    amat, bmat, cmat = approx.get_coefficients_matrix(funs, alphas, nu)
    out = approx.approx_fun_from_matrix(alphas, nu, amat, bmat, cmat, s)
    fapprox = out[:, 1]
    assert allclose(fapprox[s<alpha0], fun(alpha0))
    assert allclose(fapprox[s>alpha1], fun(alpha1))

    for alpha in alphas:
        out = approx.approx_fun_from_matrix(alphas, nu, amat, bmat, cmat, alpha)
        assert allclose(out[0, 1], fun(alpha))


def test_optimize_nu(allclose, reservoir_function):
    fname, fun, dfun, sol, inflow, (alpha0, alpha1) = reservoir_function
    funs = [lambda x: inflow+fun(x)]
    nalphas = 21
    alphas = np.linspace(alpha0, alpha1, nalphas)
    nu, amat, bmat, cmat = approx.optimize_nu(funs, alphas)

    s = np.linspace(alpha0, alpha1, 10000)
    out = approx.approx_fun_from_matrix(alphas, nu, amat, bmat, cmat, s)
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
        "recip": 1e-3, \
        "recipquad": 1e-3, \
        "runge": 1e-4, \
        "stiff": 1e-9, \
        "ratio": 5e-1
    }
    assert rmse<rmse_thresh[fname]
    LOGGER.info(f"optimize approx vs truth for {fname}: rmse={rmse:3.3e} (nu={nu:0.2f})")


def test_optimize_vs_quad(allclose, reservoir_function):
    fname, fun, dfun, sol, inflow, (alpha0, alpha1) = reservoir_function
    if fname in ["x2", "stiff"]:
        # Skip x2 and stiff which are perfect match with quadratic functions
        pytest.skip("Skip function")

    funs = [lambda x: inflow+fun(x)]
    nalphas = 21
    alphas = np.linspace(alpha0, alpha1, nalphas)
    nu, amat, bmat, cmat = approx.optimize_nu(funs, alphas)

    s = np.linspace(alpha0, alpha1, 10000)
    out = approx.approx_fun_from_matrix(alphas, nu, amat, bmat, cmat, s)
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
        "exp": 1e-7, \
        "sin": 1.0+1e-4, \
        "recip": 0.3, \
        "recipquad": 0.2, \
        "runge": 1.0+1e-4, \
        "ratio": 0.2
    }
    assert ratio<ratio_thresh[fname]
    LOGGER.info(f"optimize approx vs quad for {fname}, error ratio: {ratio:2.2e}")

