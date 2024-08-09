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
        alpha0, alpha1 = (-1., 1.)
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
        alpha0, alpha1 = (-1, 3)
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
        K, nu, alpha = 10., 0.8, 2
        fun = lambda x: alpha*(1-(x/K)**nu)*x
        dfun = lambda x: alpha*(1-(x/K)**nu)-alpha*nu*(x/K)**nu
        inflow = 0.
        sol = lambda t, s0: K/(1+((K/s0)**nu-1)*np.exp(-alpha*nu*t))**(1./nu)
        alpha0, alpha1 = 0, K


    #elif name == "cos":
    #    e = 1e-5
    #    fun = lambda x: 1+eps+np.cos(x)
    #    dfun = lambda x: -np.sin(x)
    #    inflow = 0.
    #    C = math.sqrt(e*(e+2))
    #    # int ( 1/f ) = 2/C*atan(e*sin(x)/C/(cos(x)+1))
    #    # hence
    #    # => atan(e*sin(s1)/C/(cos(s1)+1)) = atan(e*sin(s0)/C/(cos(s0)+1))+C/2.t
    #    #                                  = D+C/2.t
    #    # => e*sin(s1)/(1+cos(s1)) = tan(D+C/2.t)
    #    sol = lambda t, s0: (1-np.exp(-n*t)*(1-s0**n))**(1./n)
    #    alpha0, alpha1 = 0., 3*math.pi


    return name, fun, dfun, sol, inflow, (alpha0, alpha1)


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
        name = "Constraint b=-c"
        params[:, 2] = -params[:, 1]
    elif case == 9:
        name = "Constraint b=c"
        params[:, 2] = params[:, 1]
    elif case ==10:
        name = "General"
    elif case ==11:
        name = "General+large scale"
        params *= 1000
    elif case ==12:
        name = "General+low scale"
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

    return f"{name:20}", case, params, nus, s0s, Tmax



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
        o = approx.approx_fun(nu, a, b, c, s0s)
        expected = a+b*np.exp(-nu*s0s)+c*np.exp(nu*s0s)
        assert allclose(o, expected)

        o_slow = [slow.approx_fun(nu, a, b, c, s) for s in s0s]
        assert allclose(o_slow, expected)

    a, b, c = [np.ascontiguousarray(v) for v in params.T]
    o = approx.approx_fun(nu, a, b, c, s0s)
    expected = a+b*np.exp(-nu*s0s)+c*np.exp(nu*s0s)
    assert allclose(o, expected)

    # Check derivative against nu
    do = approx.approx_fun(nu, 0, -s0s*b, s0s*c, s0s)
    eps = 1e-8
    oe = approx.approx_fun(nu+eps, a, b, c, s0s)
    expected = (oe-o)/eps
    assert allclose(do, expected, atol=1e-4)


def test_get_coefficients(allclose, reservoir_function):
    fname, fun, dfun, _, _, (alpha0, alpha1) = reservoir_function
    nus = [0.01, 0.1, 1, 5]

    #nus = [5]
    # check fun = genlogisitc

    corrected = 0
    for nu in nus:
        a, b, c, corr = approx.get_coefficients(fun, alpha0, alpha1, nu)
        corrected += corr

        # Check function on bounds
        fa = approx.approx_fun(nu, a, b, c, alpha0)
        ft = fun(alpha0)
        assert allclose(ft, fa, atol=1e-7)

        fa = approx.approx_fun(nu, a, b, c, alpha1)
        ft = fun(alpha1)
        assert allclose(ft, fa, atol=1e-7)

        # Check mid point values
        if not corrected:
            eps = 0.5
            x = (1-eps)*alpha0+eps*alpha1
            assert allclose(approx.approx_fun(nu, a, b, c, x), fun(x))

    LOGGER.info("")
    LOGGER.info(f"[{fname}] get coeff: corrected = {corrected}/{len(nus)}")


def test_get_coefficients_edge_cases(allclose):
    # Piecewise linear fun
    x0, xm, x1 = 0., 0.5, 1.
    def fun(x, f0, fm, f1):
        scal = np.isscalar(x)
        x = np.atleast_1d(x)
        y = f0*np.ones_like(x)
        y = np.where((x<xm)&(x>=x0), f0+(fm-f0)*(x-x0)/(xm-x0), y)
        y = np.where((x>=xm)&(x<x1), fm+(f1-fm)*(x-xm)/(x1-xm), y)
        y = np.where(x>=x1, f1, y)
        if scal:
            return y[0]
        else:
            return y

    nu = 1.

    # Case of a slowly varying monotonous function
    f = lambda x: fun(x, 0, 1, 2)
    a, b, c, corr = approx.get_coefficients(f, x0, x1, nu)
    assert not corr

    # Case of non-monotonous function
    f = lambda x: fun(x, 0, 6, 2)
    a, b, c, corr = approx.get_coefficients(f, x0, x1, nu)
    assert not corr

    # Case of a highly varying function
    f = lambda x: fun(x, 0, 2, 2)
    a, b, c, corr = approx.get_coefficients(f, x0, x1, nu)
    assert corr


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

    # ratio function cannot be captured with 11 bands
    # increase this to 31
    nalphas = 31 if fname == "ratio" else 11

    alphas = np.linspace(alpha0, alpha1, nalphas)
    scr = np.ones(len(funs))
    nu, amat, bmat, cmat, niter, fopt = approx.optimize_nu(funs, alphas, scr)
    assert approx.is_continuous(alphas, nu, amat, bmat, cmat)

    s = np.linspace(alpha0, alpha1, 10000)
    out = approx.approx_fun_from_matrix(alphas, nu, amat, bmat, cmat, s)
    fapprox = out[:, 0]
    ftrue = funs[0](s)
    rmse = math.sqrt(((fapprox-ftrue)**2).mean())

    rmse_thresh = {
        "x2": 1e-9, \
        "x4": 1e-4, \
        "x6": 1e-3, \
        "x8": 1e-3, \
        "tanh": 5e-2, \
        "exp": 1e-7,
        "sin": 5e-3, \
        "recip": 5e-3, \
        "recipquad": 5e-2, \
        "runge": 1e-3, \
        "stiff": 1e-7, \
        "ratio": 0.8, \
        "logistic": 1e-8, \
        "genlogistic": 5e-3
    }

    assert rmse<rmse_thresh[fname]
    LOGGER.info("")
    LOGGER.info(f"[{fname}] optimize approx vs truth: "\
                    +f"nalphas={nalphas} niter={niter} rmse={rmse:3.3e} (nu={nu:0.2f})")


def test_optimize_vs_quad(allclose, reservoir_function):
    fname, fun, dfun, sol, inflow, (alpha0, alpha1) = reservoir_function
    if fname in ["x2", "stiff", "logistic"]:
        # Skip x2, stiff and logistic which are perfect match with quadratic functions
        pytest.skip("Skip function")

    funs = [lambda x: inflow+fun(x)]
    nalphas = 21
    alphas = np.linspace(alpha0, alpha1, nalphas)
    scr = np.ones(len(funs))
    nu, amat, bmat, cmat, _, _ = approx.optimize_nu(funs, alphas, scr)
    assert approx.is_continuous(alphas, nu, amat, bmat, cmat)

    # Issue with continuity for fname=recip and
    # nu in [3.01, 3.03] -> TOFIX!

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
    ratio_thresh = {
        "x4": 0.5, \
        "x6": 0.3, \
        "x8": 0.2, \
        "tanh": 1.0+1e-4, \
        "exp": 1e-6, \
        "sin": 1.0+1e-4, \
        "recip": 0.7, \
        "recipquad": 0.9, \
        "runge": 1.0+1e-4, \
        "ratio": 0.9, \
        "genlogistic": 0.9
    }
    ratio = rmse/rmse_quad
    assert ratio<ratio_thresh[fname]
    LOGGER.info("")
    LOGGER.info(f"[{fname}] optimize approx vs quad: error ratio={ratio:2.2e}")


