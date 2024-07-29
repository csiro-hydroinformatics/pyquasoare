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
from pyrezeq import rezeq, rezeq_slow
import c_pyrezeq

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent

NCASES = 15
PARAM_MAX = 5
S0_MAX = 5
NU_MAX = 5

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

    v0, v1 = -PARAM_MAX*np.ones(3), PARAM_MAX*np.ones(3)
    eps = np.random.uniform(0, 1, size=(ntry, 3))

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

    params = v0[None, :]+(v1-v0)[None, :]*eps

    eps = rezeq.REZEQ_EPS
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
    nus = np.random.uniform(0, NU_MAX, ntry)
    s0s = np.random.uniform(-S0_MAX, S0_MAX, ntry)
    Tmax = 20

    return name, case, params, nus, s0s, Tmax


# ----- UTILITY FUNCTIONS ------------------------------------------
def get_data():
    fq = FTEST / "streamflow_423202_423206_event.csv"
    data, _ = csv.read_csv(fq, index_col=0, parse_dates=True)
    data = data.apply(lambda x: x.interpolate())
    inflows = data.FLOWUP
    outflows = data.FLOWDOWN
    return inflows, outflows

def get_nprint(ntry):
    return 50

def plot_solution(t, s1, expected=None, title="", params=None, \
                        s0=None, clear=False, show=False):
    if clear:
        plt.close("all")
    fig, ax = plt.subplots()

    ax.plot(t, s1, label="Analytical", color="tab:blue")
    if not s0 is None:
        ax.plot(t[0], s0, "o", label="Initial", color="tab:red", ms=8)

    if not expected is None:
        ax.plot(t, expected, lw=3, label="Numerical", \
                    zorder=-10, color="tab:orange")

        err = np.abs(s1-expected)
        tax = ax.twinx()
        tax.plot(t, err, "k--", lw=0.9)
        lab = f"Error max={err.max():3.3e}"
        ax.plot([], [], "k--", lw=0.9, label=lab)

    ax.legend()

    if not params is None:
        nu, a, b, c = params
        title = f"{title} - nu={nu:0.2f} a={a:0.2f} b={b:0.2f} c={c:0.2f}"
    ax.set(title=title, xlabel="time", ylabel="value")

    if show:
        plt.show()

    return ax


# ----- TEST FUNCTIONS --------------------------------------------

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
    nprint = get_nprint(ntry)

    for itry, ((a, b, c), nu, s0) in enumerate(zip(params, nus, s0s)):
        ds = rezeq.approx_fun(nu, a, b, c, s0s)
        expected = a+b*np.exp(-nu*s0s)+c*np.exp(nu*s0s)
        assert allclose(ds, expected)

        ds_slow = [rezeq_slow.approx_fun(nu, a, b, c, s) for s in s0s]
        assert allclose(ds_slow, expected)

        jac = rezeq.approx_jac(nu, a, b, c, s0s)
        expected = -nu*b*np.exp(-nu*s0s)+nu*c*np.exp(nu*s0s)
        assert allclose(jac, expected)

        jac_slow = [rezeq_slow.approx_jac(nu, a, b, c, s) for s in s0s]
        assert allclose(jac_slow, expected)


def test_integrate_delta_t_max(allclose, generate_samples, printout):
    cname, case, params, nus, s0s, Tmax = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("Skip param config")
    nprint = get_nprint(ntry)
    t_eval = np.linspace(0, Tmax, 1000)

    print("")
    print(" "*4+f"Testing integrate_tmax - case {case} / {cname}")
    err_max = 0
    nskipped = 0
    ndpos = 0
    ndelta = 0
    for itry, ((a, b, c), nu, s0) in enumerate(zip(params, nus, s0s)):

        # Log progress
        if itry%nprint==0 and printout:
            print(" "*8+f"tmax - case {case} - Try {itry+1:4d}/{ntry:4d}")

        # Run solver first to see how far it goes
        t0 = 0
        f = lambda x: rezeq.approx_fun(nu, a, b, c, x)
        df = lambda x: rezeq.approx_jac(nu, a, b, c, x)
        te, ns1, nev, njac = rezeq_slow.integrate_forward_numerical(\
                                    [f], [df], t0, [s0], t_eval)

        # Check tmax < end of sim
        if te.max()<Tmax and te.max()>0 and len(te)>3:
            Delta = a**2-4*b*c
            ndpos += Delta>0

            # Refines
            t0, t1 = te[[-3, -1]]
            te = np.linspace(t0, 2*t1-t0, 1000)
            s0 = ns1[-3]
            te, ns1, nev, njac = rezeq_slow.integrate_forward_numerical(\
                                                    [f], [df], t0, [s0], te)
            expected = te.max()

            s1 = rezeq.integrate_forward(nu, a, b, c, t0, s0, te)
            dtm = rezeq.integrate_delta_t_max(nu, a, b, c, s0)
            assert dtm>0
            tm = t0+dtm

            err = abs(np.log(tm)-np.log(expected))
            assert err<2e-3
            #plot_solution(te, s1, ns1, show=True, params=[nu, a, b, c])
            err_max = max(err, err_max)

            dtm_slow = rezeq_slow.integrate_delta_t_max(nu, a, b, c, s0)
            assert np.isclose(dtm, dtm_slow)

        else:
            nskipped += 1

    mess = " "*4+f">> Errmax = {err_max:3.2e}"\
            f"  skipped={100*nskipped/ntry:0.0f}%"
    if case>=9:
        if nskipped<ntry:
            mess += f" - Delta>0={100*ndpos/(ntry-nskipped):0.0f}%"

    print(mess)
    print("")


def test_steady_state(allclose, generate_samples):
    cname, case, params, nus, _, _ = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("Skip param config")
    nprint = get_nprint(ntry)

    print("")
    print(" "*4+f"Testing steady state - case {case} / {cname}")
    err_max = 0
    a, b, c = [np.ascontiguousarray(v) for v in params.T]
    steady = rezeq.steady_state(nus, a, b, c)

    if case<4:
        # No steady state
        assert np.all(np.isnan(steady))
        pytest.skip("No steady state for this case")

    if np.all(np.isnan(steady)):
        pytest.skip("No steady state found")

    # check nan values
    iboth = np.isnan(steady).sum(axis=1)==0
    assert np.all(np.diff(steady[iboth], axis=1)>=0)

    if case>=8:
        # 2 distinct roots
        Delta = a**2-4*b*c
        ipos = Delta>0
        assert np.all(np.diff(steady[iboth&ipos], axis=1)>0)

    ione = np.isnan(steady).sum(axis=1)==1
    assert np.all(~np.isnan(steady[ione, 0]))

    # check steady state
    f = np.array([[rezeq.approx_fun(nu, aa, bb, cc, s) for nu, aa, bb, cc, s\
                        in zip(nus, a, b, c, ss)] for ss in steady.T]).T
    err_max = np.nanmax(np.abs(f))
    assert err_max < 5e-4

    nskipped = np.all(np.isnan(f), axis=1).sum()
    mess = " "*4+f">> Errmax = {err_max:3.2e}"\
            f"  skipped={100*(nskipped)/ntry:0.0f}%"
    print(mess)
    print("")


def test_integrate_forward_vs_finite_difference(allclose, generate_samples, printout):
    cname, case, params, nus, s0s, Tmax = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()
    nprint = get_nprint(ntry)

    print("")
    print(" "*4+f"Testing integrate_forward using diff - case {case} / {cname}")
    t0 = 0
    errmax_max = 0
    notskipped = 0
    ndelta = 0
    for itry, ((a, b, c), nu, s0) in enumerate(zip(params, nus, s0s)):
        # Log progress
        if itry%nprint==0 and printout:
            print(" "*8+f"forward - case {case} - Try {itry+1:4d}/{ntry:4d}")

        # Set integration time
        Tmax = min(20, t0+rezeq.integrate_delta_t_max(nu, a, b, c, s0)*0.99)
        if np.isnan(Tmax) or Tmax<0:
            continue
        t_eval = np.linspace(0, Tmax, 1000)

        s1 = rezeq.integrate_forward(nu, a, b, c, t0, s0, t_eval)
        if np.all(np.isnan(s1)):
            continue

        Tmax_rev = t_eval[~np.isnan(s1)].max()
        if Tmax_rev<1e-10:
            continue

        t_eval_rev = np.linspace(0, Tmax_rev, 10000)
        s1 = rezeq.integrate_forward(nu, a, b, c, t0, s0, t_eval_rev)

        # Compare with slow
        s1_slow = [rezeq_slow.integrate_forward(nu, a, b, c, t0, s0, t) \
                            for t in t_eval_rev]
        assert allclose(s1, s1_slow)

        # Test if s1 is monotone
        ds1 = np.round(np.diff(s1), decimals=6)
        ads1 = np.abs(ds1)
        sgn = np.sign(ds1)
        sgn_ref = np.sign(ds1[np.argmax(ads1)])
        assert np.all(sgn[ads1>1e-8]==sgn_ref)

        # Differentiate using 5th point method
        h = t_eval_rev[1]-t_eval_rev[0]
        ds1 = (-s1[4:]+8*s1[3:-1]-8*s1[1:-3]+s1[:-4])/h/12
        td = t_eval_rev[2:-2]

        expected = rezeq.approx_fun(nu, a, b, c, s1[2:-2])

        err = np.abs(np.arcsinh(ds1*1e-3)-np.arcsinh(expected*1e-3))
        iok = (np.abs(ds1)<1e2) & (td>td[2]) & (td<td[-2])
        if iok.sum()<4:
            continue

        errmax = np.nanmax(err[iok])
        notskipped += 1
        assert errmax<5e-4

        errmax_max = max(errmax, errmax_max)

    print(" "*4+f">> Errmax = {errmax_max:3.2e}"\
                f" skipped={(ntry-notskipped)*100/ntry:0.0f}%")
    print("")


def test_integrate_forward_vs_numerical(allclose, generate_samples, printout):
    cname, case, params, nus, s0s, Tmax = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()
    nprint = get_nprint(ntry)

    print("")
    print(" "*4+f"Testing integrate_forward vs numerical - case {case} / {cname}")
    t0 = 0
    errmax_max = 0
    perc_delta_pos = 0
    nskipped = 0
    ndelta = 0
    for itry, ((a, b, c), nu, s0) in enumerate(zip(params, nus, s0s)):
        # Count the frequence of positive determinant
        delta = a**2-4*b*c
        if abs(delta)>0:
            ndelta += 1
            perc_delta_pos += delta>0

        # Log progress
        if itry%nprint==0 and printout:
            print(" "*8+f"forward - {name} - Try {itry+1:4d}/{ntry:4d}")

        # Set integration time
        Tmax = min(20, t0+rezeq.integrate_delta_t_max(nu, a, b, c, s0)*0.99)
        if np.isnan(Tmax) or Tmax<0:
            continue
        t_eval = np.linspace(0, Tmax, 1000)

        f = lambda x: rezeq.approx_fun(nu, a, b, c, x)
        df = lambda x: rezeq.approx_jac(nu, a, b, c, x)
        te, expected, nev, njac = rezeq_slow.integrate_forward_numerical([f], [df], \
                                                            t0, [s0], t_eval)
        if len(te)<3:
            nskipped += 1
            continue

        dsdt = np.diff(expected)/np.diff(te)
        dsdt = np.insert(dsdt, 0, 0)

        s1 = rezeq.integrate_forward(nu, a, b, c, t0, s0, te)

        err = np.abs(np.arcsinh(s1*1e-3)-np.arcsinh(expected*1e-3))
        iok = np.abs(dsdt)<1e3
        errmax = np.nanmax(err[iok])
        assert errmax<1e-4
        errmax_max = max(errmax, errmax_max)

    perc_skipped = nskipped*100/ntry
    mess = f"Errmax = {errmax_max:3.2e}  Skipped={perc_skipped:0.0f}%"
    if case>=8:
        perc_delta_pos = perc_delta_pos/ndelta*100
        mess += f"  Delta>0 = {perc_delta_pos:0.0f}%"
    print(" "*4+mess)
    print("")


def test_integrate_inverse(allclose, generate_samples, printout):
    cname, case, params, nus, s0s, Tmax = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()
    nprint = get_nprint(ntry)

    print("")
    print(" "*4+f"Testing integrate_inverse - case {case} / {cname}")
    t0 = 0
    nskipped = 0
    errmax_max = 0
    for itry, ((a, b, c), nu, s0) in enumerate(zip(params, nus, s0s)):
        if itry%nprint==0 and printout:
            print(" "*8+f"inverse - case {case} - Try {itry+1:4d}/{ntry:4d}")

        # Set integration time
        Tmax = min(20, t0+rezeq.integrate_delta_t_max(nu, a, b, c, s0)*0.99)
        if np.isnan(Tmax) or Tmax<0:
            nskipped += 1
            continue
        t_eval = np.linspace(0, Tmax, 1000)

        # Simulate
        s1 = rezeq.integrate_forward(nu, a, b, c, t0, s0, t_eval)
        ds1 = rezeq.approx_fun(nu, a, b, c, s1)

        iok = ~np.isnan(s1) & (ds1>1e-5)
        iok[0] = False
        if iok.sum()<5:
            nskipped += 1
            continue

        t, s1 = t_eval[iok], s1[iok]

        # Compute difference
        ta = rezeq.integrate_inverse(nu, a, b, c, s0, s1)
        assert np.all(ta>=0)

        dsdt = rezeq.approx_fun(nu, a, b, c, s1)
        err = np.abs(np.log(ta*1e-3)-np.log((t-t0)*1e-3))
        iok = (dsdt>1e-4) & (dsdt<1e4)
        if iok.sum()<5:
            nskipped += 1
            continue

        errmax = np.nanmax(err[iok])
        assert errmax< 5e-6 if case in [9, 12] else 1e-8
        errmax_max = max(errmax, errmax_max)

        # Compare with slow
        ta_slow = [rezeq_slow.integrate_inverse(nu, a, b, c, s0, s) for s in s1]
        assert allclose(ta, ta_slow)

    perc_skipped = nskipped*100/ntry
    mess = f"Errmax = {errmax_max:3.2e}  Skipped={perc_skipped:0.0f}%"
    print(" "*4+mess)
    print("")


def test_get_coefficients(allclose, reservoir_function):
    fname, fun, dfun, _, _, (alpha0, alpha1) = reservoir_function
    nus = [0.01, 0.1, 1, 5]

    for eps, nu in prod([0.2, 0.5, 0.8], nus):
        a, b, c = rezeq.get_coefficients(fun, alpha0, alpha1, nu, eps)

        # Check function on bounds
        assert allclose(rezeq.approx_fun(nu, a, b, c, alpha0), fun(alpha0), atol=1e-7)
        assert allclose(rezeq.approx_fun(nu, a, b, c, alpha1), fun(alpha1), atol=1e-7)

        # Check mid point values
        x = (1-eps)*alpha0+eps*alpha1
        assert allclose(rezeq.approx_fun(nu, a, b, c, x), fun(x))


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
    nf = rezeq.REZEQ_NFLUXES_MAX
    fs = [lambda x: x**a for a in np.linspace(1, 2, nf+1)]
    with pytest.raises(ValueError, match="nfluxes"):
        _, amat, bmat, cmat, emat = rezeq.get_coefficients_matrix(fs, \
                                                        alphas, n, eps)

    amat, bmat, cmat = rezeq.get_coefficients_matrix(funs, \
                                                alphas, nu, epsilon)
    out = rezeq.approx_fun_from_matrix(alphas, nu, amat, bmat, cmat, s)
    fapprox = out[:, 1]
    assert allclose(fapprox[s<alpha0], fun(alpha0))
    assert allclose(fapprox[s>alpha1], fun(alpha1))

    for alpha in alphas:
        out = rezeq.approx_fun_from_matrix(alphas, nu, amat, bmat, cmat, alpha)
        assert allclose(out[0, 1], fun(alpha))


def test_steady_state_scalings(allclose):
    nalphas = 500
    alphas = np.linspace(0, 1.2, nalphas)
    # GR4J production
    funs = [
        lambda x: 1-x**2, \
        lambda x: -x*(2-x), \
        lambda x: -x**4/3
    ]
    nu = 1
    amat, bmat, cmat = rezeq.get_coefficients_matrix(funs, alphas, nu)

    nval = 200
    scalings = np.random.uniform(0, 100, size=(nval, 3))
    scalings[:, -1] = 1

    steady = rezeq.steady_state_scalings(alphas, nu, scalings, amat, bmat, cmat)
    for t in range(nval):
        s0 = steady[t]
        # Check steady on approx fun
        amats = amat*scalings[t][None, :]
        bmats = bmat*scalings[t][None, :]
        cmats = cmat*scalings[t][None, :]
        out = rezeq.approx_fun_from_matrix(alphas, nu, amats, bmats, cmats, s0)
        fsum = out.sum(axis=1)
        assert allclose(fsum[~np.isnan(fsum)], 0.)

        # Check steady on original fun
        feval = np.array([f(s0)*scalings[t, ifun] for ifun, f in enumerate(funs)])
        fsum = np.sum(feval, axis=0)
        assert allclose(fsum[~np.isnan(fsum)], 0., atol=1e-7)


def test_find_alphas(allclose):
    alphas = np.linspace(0, 1, 4)
    u0 = -1.
    ialpha = rezeq.find_alpha(alphas, u0)
    assert ialpha == 0

    u0 = 1.1
    ialpha = rezeq.find_alpha(alphas, u0)
    assert ialpha == 2

    u0 = 0.2
    ialpha = rezeq.find_alpha(alphas, u0)
    assert ialpha == 0

    u0 = 0.4
    ialpha = rezeq.find_alpha(alphas, u0)
    assert ialpha == 1

    u0 = 0.7
    ialpha = rezeq.find_alpha(alphas, u0)
    assert ialpha == 2


def test_increment_fluxes_vs_integration(allclose, \
                                        generate_samples, printout):
    cname, case, params, nus, s0s, Tmax = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()
    nprint = get_nprint(ntry)

    # Reduce range of nu
    nus *= 1.5/NU_MAX

    print("")
    print(" "*4+f"Testing increment_fluxes - numerical integration"+\
                f" - case {case} / {cname}")
    nskipped = 0
    errbal_max = 0
    errmax_max = 0
    ev = []
    for itry, ((aoj, boj, coj), nu, s0) in enumerate(zip(params, nus, s0s)):
        if itry%nprint==0 and printout:
            print(" "*8+f"flux integration - case {case} - Try {itry+1:4d}/{ntry:4d}")

        avect, bvect, cvect = np.random.uniform(-1, 1, size=(3, 3))

        # make sure coefficient sum matches aoj, boj and coj
        sa, sb, sc = avect.sum(), bvect.sum(), cvect.sum()
        avect += (aoj-sa)/3
        bvect += (boj-sb)/3
        cvect += (coj-sc)/3

        # Integrate forward analytically
        t1 = min(10, rezeq.integrate_delta_t_max(nu, aoj, boj, coj, s0))
        t0 = t1*0.05 # do not start at zero to avoid sharp falls
        t1 = t1*0.5 # far away from limits of validity
        s1 = rezeq.integrate_forward(nu, aoj, boj, coj, t0, s0, t1)
        if np.isnan(s1):
            nskipped += 1
            continue

        # Check error if sum of coefs is not matched
        with pytest.raises(ValueError):
            cvect2 = cvect.copy()
            cvect2[0] += 10
            fluxes = np.zeros(3)
            rezeq.increment_fluxes(nu, avect, bvect, cvect2, \
                            aoj, boj, coj, t0, t1, s0, s1, fluxes)

        # Compute fluxes analytically
        fluxes = np.zeros(3)
        rezeq.increment_fluxes(nu, avect, bvect, cvect, \
                        aoj, boj, coj, t0, t1, s0, s1, fluxes)

        a, b, c = aoj, boj, coj
        Delta = a**2-4*b*c

        # Test mass balance
        balance = s1-s0-fluxes.sum()
        errbal_max = max(abs(balance), errbal_max)
        #assert allclose(balance, 0)

        # Compare against numerical integration
        def finteg(t, a, b, c):
            s = rezeq.integrate_forward(nu, aoj, boj, coj, t0, s0, t)
            return rezeq.approx_fun(nu, a, b, c, s)

        expected = np.array([integrate.quad(finteg, t0, t1, args=(a, b, c))\
                        for a, b, c in zip(avect, bvect, cvect)])
        tol = expected[:, 1].max()
        errmax = max(abs(fluxes-expected[:, 0]))

        Delta = aoj**2-4*boj*coj
        w = nu*math.sqrt(abs(Delta))/2*(t1-t0)
        # integration not trusted when overflow becomes really bad
        if w<1000:
            errmax_max = max(errmax, errmax_max)

        # Compare against slow
        fluxes_slow = np.zeros(3)
        rezeq_slow.increment_fluxes(nu, avect, bvect, cvect, \
                        aoj, boj, coj, t0, t1, s0, s1, fluxes_slow)
        assert allclose(fluxes, fluxes_slow)


    # We should not skip any simulation because we stop at t=tmax
    #assert nskipped == 0
    mess = f"Errmax = {errmax_max:3.2e} ({ntry-nskipped} runs) "+\
                f" Balmax = {errbal_max:3.3e}"
    print(" "*4+mess)
    print("")


def test_integrate_reservoir_equation_extrapolation(allclose, reservoir_function):
    fname, fun, dfun, _, inflow, (alpha0, alpha1) = reservoir_function

    # Reservoir functions
    inp = lambda x: inflow
    sfun = lambda x: inflow+fun(x)
    funs = [inp, fun]

    # Optimize nu
    nalphas = 5
    alphas, nus = rezeq.get_coefficients_matrix_optimize_nu(funs, \
                                    alpha0, alpha1, nalphas)

    print("")
    print(" "*4+f"Testing rezeq integrate extrapolation - fun {fname} ")

    # Approximate coefficients
    nus, amat, bmat, cmat, _ = rezeq.get_coefficients_matrix(funs, alphas, nus=nus)

    t0 = 0 # Analytical solution always integrated from t0=0!
    nval = 500
    Tmax = 5
    t1 = np.linspace(t0, Tmax, nval)

    # Initial conditions outside of approximation range [alpha0, alpha1]
    dalpha = alpha1-alpha0
    s0_high = alpha1+dalpha*1.5
    s0_low = alpha0-dalpha*1.5

    # Approximate method
    scalings = np.ones(2)
    approx_low, approx_high = [s0_low], [s0_high]
    s_start_low = s0_low
    s_start_high = s0_high

    for i in range(len(t1)-1):
        t_start = t1[i]
        delta = t1[i+1]-t_start
        # integrate when below and above approximation range
        _, s_end_low, _ = rezeq.integrate(alphas, scalings, nus, \
                                            amat, bmat, cmat, t_start, \
                                            s_start_low, delta)
        # .. compare against slow
        _, s_end_low_slow, _ = rezeq_slow.integrate(alphas, scalings, nus, \
                                            amat, bmat, cmat, t_start, \
                                            s_start_low, delta)
        assert np.isclose(s_end_low, s_end_low_slow)
        approx_low.append(s_end_low)
        s_start_low = s_end_low

        _, s_end_high, _ = rezeq.integrate(alphas, scalings, nus, \
                                            amat, bmat, cmat, t_start, \
                                            s_start_high, delta)
        # .. compare against slow
        _, s_end_high_slow, _ = rezeq_slow.integrate(alphas, scalings, nus, \
                                            amat, bmat, cmat, t_start, \
                                            s_start_high, delta)
        assert np.isclose(s_end_high, s_end_high_slow)
        approx_high.append(s_end_high)
        s_start_high = s_end_high

    # Expect constant derivative of solution in extrapolation mode
    approx_low = np.array(approx_low)
    ilow = approx_low<alpha0
    assert ilow.sum()>0
    dt = t1[1]-t1[0]
    ds_low = np.diff(approx_low[ilow])/dt
    expected_low = [rezeq.approx_fun(nus[0], amat[0, i], bmat[0, i], \
                    cmat[0, i], alphas[0]) for i in range(2)]
    expected_low = np.array(expected_low).sum()
    assert allclose(ds_low, expected_low)

    approx_high = np.array(approx_high)
    ihigh = approx_high>alpha1
    assert ihigh.sum()>0
    ds_high = np.diff(approx_high[ihigh])/dt
    expected_high = [rezeq.approx_fun(nus[-1], amat[-1, i], bmat[-1, i], \
                    cmat[-1, i], alphas[-1]) for i in range(2)]
    expected_high = np.array(expected_high).sum()
    assert allclose(ds_high, expected_high)



def test_integrate_reservoir_equation(allclose, ntry, reservoir_function):
    fname, fun, dfun, sol, inflow, (alpha0, alpha1) = reservoir_function
    if sol is None:
        pytest.skip("No analytical solution")

    #pytest.skip("WIP")
    inp = lambda x: inflow
    sfun = lambda x: inflow+fun(x)
    funs = [sfun, inp, fun]

    dinp = lambda x: 0.
    dsfun = lambda x: dinp(x)+dfun(x)
    dfuns = [dsfun, dinp, dfun]

    # Optimize nu
    nalphas = 5
    epsilons = 1e-6
    alphas, nus = rezeq.get_coefficients_matrix_optimize_nu([inp, fun], \
                                    alpha0, alpha1, nalphas, \
                                    epsilons=epsilons)

    print("")
    print(" "*4+f"Testing rezeq integrate - fun {fname} "\
                +f"ntry={ntry} nu={nus[0]:0.2f}")

    # Approximate coefficients
    nus, amat, bmat, cmat, _ = rezeq.get_coefficients_matrix([inp, fun], \
                                                                alphas, \
                                                                nus=nus, \
                                                                epsilons=epsilons)
    scalings = np.ones(2)

    t0 = 0 # Analytical solution always integrated from t0=0!
    nval = 100
    Tmax = 100
    t1 = np.linspace(t0, Tmax, nval)

    errmax_app_max, time_app, niter_app = 0., 0., 0
    errmax_num_max, time_num, niter_num = 0., 0., 0

    for itry in range(ntry):
        if fname == "runge":
            s0 = np.random.uniform(-1, 1)
        else:
            s0 = np.random.uniform(alpha0, alpha1)

        s0 = 0.5

        # Analytical solution
        expected = sol(t1, s0)

        # Approximate method
        niter, approx = [0], [s0]
        s_start = s0
        start_exec = time.time()
        for i in range(len(t1)-1):
            t_start = t1[i]
            #print(f"\n### [{i:3d}] t_start = {t_start} / s_start = {s_start:0.5f}###")
            delta = t1[i+1]-t_start
            n, s_end, _ = rezeq.integrate(alphas, scalings, nus, \
                                                amat, bmat, cmat, t_start, \
                                                s_start, delta)
            # Against slow
            #n_slow, s_end_slow, _ = rezeq_slow.integrate(alphas, scalings, nus, \
            #                                    amat, bmat, cmat, t_start, \
            #                                    s_start, delta)
            #assert np.isclose(s_end, s_end_slow)

            niter.append(n)
            approx.append(s_end)
            s_start = s_end

        #print("\n######### Finito ##########")

        end_exec = time.time()
        time_app += (end_exec-start_exec)*1e3
        niter = np.array(niter)
        approx = np.array(approx)
        niter_app = max(niter_app, niter.sum())

        # Numerical method
        start_exec = time.time()
        tn, fn, nev, njac = rezeq_slow.integrate_forward_numerical(\
                                        funs, dfuns, \
                                        t0, [s0]+[0]*2, t1)
        end_exec = time.time()
        time_num += (end_exec-start_exec)*1e3
        numerical = fn[:, 0]
        niter_num = max(niter_num, nev+njac)

        # Errors
        errmax = np.abs(approx-expected).max()
        if errmax>errmax_app_max:
            errmax_app_max = errmax
            s0_errmax = s0

        errmax = np.abs(numerical-expected).max()
        errmax_num_max = max(errmax, errmax_num_max)

    #assert errmax_app_max<1e-3

    if False:
        import matplotlib.pyplot as plt
        expected = sol(t1, s0_errmax)

        s_start = s0
        approx = [s0]
        for i in range(len(t1)-1):
            start = t1[i]
            delta = t1[i+1]-start
            n, s_end, _ = rezeq.integrate(alphas, scalings, nus, \
                                                amat, bmat, cmat, start, \
                                                s_start, delta)
            approx.append(s_end)
            s_start = s_end

        approx = np.array(approx)

        from hydrodiy.plot import putils

        plt.close("all")
        fig, axs = plt.subplots(ncols=2)

        ax = axs[0]
        xx = np.linspace(alpha0, 0.5, 500)
        ds = sfun(xx)
        ax.plot(ds, xx, label="expected", color="tab:blue")

        ds = sfun(expected)
        ax.plot(ds, expected, lw=4, label="expected sim", color="tab:blue")
        ax.plot(ds[0], expected[0], "x", ms=5, color="tab:blue")

        ds = rezeq.approx_fun_from_matrix(alphas, nus, \
                                        amat, bmat, cmat, xx)
        ds = ds.sum(axis=1)
        ax.plot(ds, xx, label="approx", color="tab:orange")

        ds = rezeq.approx_fun_from_matrix(alphas, nus, \
                                        amat, bmat, cmat, approx)
        ds = ds.sum(axis=1)
        ax.plot(ds, approx, lw=4, label="expected sim", color="tab:orange")
        ax.plot(ds[0], approx[0], "x", ms=5, color="tab:orange")
        putils.line(ax, 0, 1, 0, 0, "k-", lw=0.8)

        #for j in range(nalphas-1):
        #    nu, a, b, c = nus[j], amat.sum(axis=1)[j], bmat.sum(axis=1)[j], cmat.sum(axis=1)[j]
        #    ds = rezeq.approx_fun(nu, a, b, c, xx)
        #    ax.plot(ds, xx, label=f"approx {j}")

        ax.legend()

        ax = axs[1]
        ax.plot(t1, expected, label="expected")
        ax.plot(t1, approx, label="approx")
        ax.legend()
        ax.set(title=f"fun={fname} s0={s0_errmax:0.5f}")
        plt.show()
        import pdb; pdb.set_trace()


    tab = " "*8
    print(f"{tab}approx vs analytical = {errmax_app_max:3.2e}"\
                    +f" / time={time_app:3.3e}ms / niter={niter_app}")
    print(f"{tab}numer  vs analytical = {errmax_num_max:3.2e}"\
                    +f" / time={time_num:3.3e}ms / niter={niter_num}")
    print("")

