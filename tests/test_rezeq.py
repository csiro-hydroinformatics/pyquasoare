from pathlib import Path
import math
import re
import pytest
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from itertools import product as prod

import time

import warnings

from hydrodiy.io import csv
from pyrezeq import rezeq

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent

NCASES = 10
PARAM_MAX = 5
S0_MAX = 5
NU_MAX = 5

# ----- FIXTURES ---------------------------------------------------
@pytest.fixture(scope="module", \
            params=["x2", "x5", "tanh", "exp", "sin", "recip", "recipquad"])
def reservoir_function(request):
    name = request.param
    # sol is the analytical solution of ds/dt = 1+fun(s)
    if name == "x2":
        sol = lambda t, s0: (s0+np.tanh(t))/(1+s0*np.tanh(t))
        return name, lambda x: -x**2, lambda x: -2*x, sol
    elif name == "x5":
        return name, lambda x: -x**5, lambda x: -5*x**4, None
    elif name == "tanh":
        return name, lambda x: -np.tanh(x), lambda x: -1+np.tanh(x)**2, None
    elif name == "exp":
        sol = lambda t, s0: s0+t-np.log(1-(1+np.exp(t))/math.exp(s0))
        return name, lambda x: -np.exp(x), lambda x: -np.exp(x), sol
    elif name == "sin":
        return name, lambda x: -1.01-np.sin(20*x), lambda x: -20*np.cos(3*x), None
    elif name == "recip":
        return name, lambda x: -1e-2/(1.01-x), lambda x: 1e-2/(1.01-x)**2, None
    elif name == "recipquad":
        return name, lambda x: -1e-4/(1.01-x)**2, lambda x: 2e-4/(1.01-x)**3, None


@pytest.fixture(scope="module", params=list(range(1, NCASES+1)))
def parameter_samples(ntry, selcase, request):
    case = request.param
    assert case in list(range(1, NCASES+1))

    if selcase>0 and case!=selcase:
        return case, np.zeros((0, 3)), "skipped"

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
    elif case == 3:
        name = "a and c are zero"
        v0[[0, 2]] = 0
        v1[[0, 2]] = 0
    elif case == 4:
        name = "a and b are zero"
        v0[[0, 1]] = 0
        v1[[0, 1]] = 0
    elif case == 5:
        name = "b is zero"
        v0[1] = 0
        v1[1] = 0
    elif case == 6:
        name = "c is zero"
        v0[2] = 0
        v1[2] = 0

    params = v0[None, :]+(v1-v0)[None, :]*eps

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

    return case, params, name


# ----- UTILITY FUNCTIONS ------------------------------------------
def get_data():
    fq = FTEST / "streamflow_423202_423206_event.csv"
    data, _ = csv.read_csv(fq, index_col=0, parse_dates=True)
    data = data.apply(lambda x: x.interpolate())
    inflows = data.FLOWUP
    outflows = data.FLOWDOWN
    return inflows, outflows

def get_nprint(ntry):
    return ntry//5

def sample_config(ntry):
    nus = np.random.uniform(0, NU_MAX, ntry)
    s0s = np.random.uniform(-S0_MAX, S0_MAX, ntry)
    Tmax = 20
    t_eval = np.linspace(0, Tmax, 5000)
    return nus, s0s, t_eval, Tmax

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

def test_approx_fun(allclose):
    abcs = np.random.uniform(-2, 2, size=(100, 3))
    nus = np.random.uniform(-2, 2, size=100)
    s = np.linspace(-10, 10)
    for abc, nu in zip(abcs, nus):
        a, b, c = abc
        ds = rezeq.approx_fun(nu, a, b, c, s)
        expected = a+b*np.exp(-nu*s)+c*np.exp(nu*s)
        assert allclose(ds, expected)

        ds = rezeq.approx_fun(nu, a, b, c, s[0])
        assert allclose(ds, expected[0])


def test_integrate_tmax(allclose, parameter_samples, printout):
    case, params, cname = parameter_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("No parameter samples")
    nprint = get_nprint(ntry)
    nus, s0s, t_eval, Tmax = sample_config(ntry)

    print("")
    print(" "*4+f"Testing integrate_tmax - case {case} / {cname}")
    err_max = 0
    ncutoff = 0
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
        te, ns1 = rezeq.integrate_forward_numerical(f, df, t0, s0, t_eval)

        if te.max()<Tmax and te.max()>0 and len(te)>3:
            ncutoff += 1
            Delta = a**2-4*b*c
            ndpos += Delta>0

            # Refines
            t0, t1 = te[[-3, -1]]
            te = np.linspace(t0, 2*t1-t0, 1000)
            s0 = ns1[-3]
            te, ns1 = rezeq.integrate_forward_numerical(f, df, t0, s0, te)
            expected = te.max()

            s1 = rezeq.integrate_forward(nu, a, b, c, t0, s0, te)
            dtm = rezeq.integrate_delta_t_max(nu, a, b, c, s0)
            assert dtm>0
            tm = t0+dtm

            err = abs(np.log(tm)-np.log(expected))
            assert err<2e-3
            err_max = max(err, err_max)

    mess = f"forward - Errmax = {err_max:3.2e}"\
            f"  cutoff={100*ncutoff/ntry:3.0f}%"
    if case in [8, 9, 10]:
        mess += f" including Delta>0={100*ndpos/ncutoff:3.0f}%"
    print(" "*8+mess)
    print("")


def test_integrate_forward_vs_finite_difference(allclose, parameter_samples, printout):
    case, params, cname = parameter_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()
    nprint = get_nprint(ntry)
    nus, s0s, t_eval, Tmax = sample_config(ntry)

    print("")
    print(" "*4+f"Testing integrate_forward using diff - case {case} / {cname}")
    t0 = 0
    errmax_max = 0
    nchecked = 0
    ndelta = 0
    for itry, ((a, b, c), nu, s0) in enumerate(zip(params, nus, s0s)):
        # Log progress
        if itry%nprint==0 and printout:
            print(" "*8+f"forward - case {case} - Try {itry+1:4d}/{ntry:4d}")

        s1 = rezeq.integrate_forward(nu, a, b, c, t0, s0, t_eval)
        Tmax_rev = t_eval[~np.isnan(s1)].max()
        t_eval_rev = np.linspace(0, Tmax_rev, 10000)
        s1 = rezeq.integrate_forward(nu, a, b, c, t0, s0, t_eval_rev)

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
        if iok.sum()<5:
            continue

        errmax = np.nanmax(err[iok])
        nchecked += 1
        assert errmax<5e-4

        errmax_max = max(errmax, errmax_max)

    mess = f"forward - Errmax = {errmax_max:3.2e}"\
                f" %checked={nchecked*100/ntry:0.0f}%"
    print(" "*8+mess)
    print("")


def test_integrate_forward_vs_solver(allclose, parameter_samples, printout):
    case, params, cname = parameter_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()
    nprint = get_nprint(ntry)
    nus, s0s, t_eval, Tmax = sample_config(ntry)

    print("")
    print(" "*4+f"Testing integrate_forward using solver - case {case} / {cname}")
    t0 = 0
    errmax_max = 0
    perc_delta_pos = 0
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

        f = lambda x: rezeq.approx_fun(nu, a, b, c, x)
        df = lambda x: rezeq.approx_jac(nu, a, b, c, x)
        te, expected = rezeq.integrate_forward_numerical(f, df, t0, s0, t_eval)
        dsdt = np.diff(expected)/np.diff(te)
        dsdt = np.insert(dsdt, 0, 0)

        s1 = rezeq.integrate_forward(nu, a, b, c, t0, s0, te)

        err = np.abs(np.arcsinh(s1*1e-3)-np.arcsinh(expected*1e-3))
        iok = np.abs(dsdt)<1e3
        errmax = np.nanmax(err[iok])
        assert errmax<1e-4

        errmax_max = max(errmax, errmax_max)

    mess = f"forward - Errmax = {errmax_max:3.2e}"
    if case in [8, 9, 10]:
        perc_delta_pos = perc_delta_pos/ndelta*100
        mess += f"   %Delta>0 = {perc_delta_pos:0.0f}%"
    print(" "*8+mess)
    print("")


def test_integrate_inverse(allclose, parameter_samples, printout):
    case, params, cname = parameter_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()
    nprint = get_nprint(ntry)
    nus, s0s, t_eval, Tmax = sample_config(ntry)

    print("")
    print(" "*4+f"Testing integrate_inverse - case {case} / {cname}")
    t0 = 0
    errmax_max = 0
    for itry, ((a, b, c), nu, s0) in enumerate(zip(params, nus, s0s)):
        if itry%nprint==0 and printout:
            print(" "*8+f"inverse - case {case} - Try {itry+1:4d}/{ntry:4d}")

        # Simulate
        s1 = rezeq.integrate_forward(nu, a, b, c, t0, s0, t_eval)

        iok = ~np.isnan(s1)
        if iok.sum()<5:
            continue

        t, s1 = t_eval[iok], s1[iok]

        # Compute difference
        ta = rezeq.integrate_inverse(nu, a, b, c, s0, s1)

        dsdt = np.diff(s1)/np.diff(t)
        dsdt = np.abs(np.insert(dsdt, 0, 0))
        err = np.abs(np.arcsinh(ta*1e-3)-np.arcsinh(t*1e-3))
        iok = (dsdt>1e-4) & (dsdt<1e4)
        if iok.sum()<5:
            continue

        errmax = np.nanmax(err[iok])
        assert errmax<(1e-3 if case==8 else 1e-10)

        errmax_max = max(errmax, errmax_max)

    print(" "*8+f"inverse - Errmax = {errmax_max:3.2e}")
    print("")


def test_get_coefficients(allclose, reservoir_function):
    # Get function and its derivative
    fname, fun, dfun, _ = reservoir_function
    alphaj = 0.
    alphajp1 = 1.
    nus = [0.01, 0.1, 1, 5]

    for eps, nu in prod([-1, 0, 0.2, 0.5, 0.8], nus):
        if eps==-1 and nu>nus[0]:
            continue
        a, b, c = rezeq.get_coefficients(fun, dfun, nu, eps, alphaj, alphajp1)

        assert allclose(rezeq.approx_fun(nu, a, b, c, alphaj), fun(alphaj))
        assert allclose(rezeq.approx_fun(nu, a, b, c, alphajp1), fun(alphajp1))

        if eps==-1:
            assert b==-c
        elif eps==0:
            assert allclose(rezeq.approx_jac(nu, a, b, c, alphaj), dfun(alphaj))

        else:
            x = (1-eps)*alphaj+eps*alphajp1
            assert allclose(rezeq.approx_fun(nu, a, b, c, x), fun(x))


def test_get_coefficients_matrix(allclose, reservoir_function):
    # Get function and its derivative
    fname, fun, dfun, _ = reservoir_function
    funs = [lambda x: 1., fun]
    dfuns = [lambda x: 0, dfun]

    nalphas = 500
    alphas = np.linspace(0., 1., nalphas)
    nus = [0.01, 1, 5]
    s = np.linspace(-0.1, 1.1, 1000)
    print("")
    errmax = {n: 0 for n in nus}
    for e, nu in prod([-1, 0, 0.2, 0.5, 0.8, 1], nus):
        if e==-1 and nu>nus[0]:
            continue
        n = nu*np.ones(nalphas-1)
        eps = e*np.ones(nalphas-1)
        ne, ae, amat, bmat, cmat = rezeq.get_coefficients_matrix(funs, dfuns, n, eps, alphas)

        out = rezeq.approx_fun_from_matrix(ae, ne, amat, bmat, cmat, s)

        fapprox = out[:, 1]
        assert allclose(fapprox[s<0], fun(0))
        assert allclose(fapprox[s>1], fun(1))

        ftrue = fun(s)
        err = fapprox-ftrue
        isin = (s>=0)&(s<=1)
        emax = np.abs(err[isin]).max()
        errmax[nu] = max(errmax[nu], emax)

        if fname in "sin":
            ethresh = 3e-2
        elif fname == "recip":
            ethresh = 1e-1
        elif fname == "recipquad":
            ethresh = 2e-1
        else:
            ethresh = 1e-5
        assert emax < ethresh

    print(f"coef matrix - fun={fname} :")
    for nu in nus:
        print(" "*4+f"errmax(nu={nu:0.2f}) = {errmax[nu]:3.3e}")



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


def test_integrate_reservoir_equations(allclose, reservoir_function):
    fname, fun, dfun, sol = reservoir_function
    if sol is None:
        pytest.skip("No analytical solution")



#def test_integrate_mass_balance(allclose):
#    deltas = np.linspace(0.1, 5, 10)
#    nalphas = 10
#    alphas = np.linspace(0, 10, nalphas)
#
#    scalings = [1.]
#    a_matrix_noscaling = [np.ones(nalphas-1)]
#    b_matrix_noscaling = [np.zeros(nalphas-1)]
#    f1 = lambda x: -(-1 if x<0 else 1)*math.sqrt(abs(x))
#    f2 = lambda x: -x
#    f3 = lambda x: -3*x**2
#    for f in [f1, f2, f3]:
#        coefs = rezeq.piecewise_linear_approximation(f, alphas)
#        scalings.append(1.)
#        a_matrix_noscaling.append(coefs[:, 0])
#        b_matrix_noscaling.append(coefs[:, 1])
#
#    u0 = 1.
#    scalings = np.array(scalings)
#    a_matrix_noscaling = np.column_stack(a_matrix_noscaling)
#    b_matrix_noscaling = np.column_stack(b_matrix_noscaling)
#
#    for delta in deltas:
#        u1, fluxes = rezeq.integrate(delta, u0, alphas, scalings, \
#                        a_matrix_noscaling, b_matrix_noscaling)
#        mass_bal = u1-u0-fluxes.sum()
#        assert allclose(mass_bal, 0.)
#
#
#def test_integrate_non_linear_reservoir(allclose):
#    delta = 1.
#
#    alpha_min = 1313
#    alpha_max = 4153
#    nalphas = 3
#    alphas = np.linspace(alpha_min, alpha_max, nalphas)
#
#    theta = 3239
#    q0 = 71.29
#    nu = 6.
#    rezfun = lambda x: -q0*(x/theta)**nu
#    coefs = rezeq.piecewise_linear_approximation(rezfun, alphas)
#
#    a_matrix_noscaling = np.column_stack([np.ones(nalphas-1), coefs[:, [0]]])
#    b_matrix_noscaling = np.column_stack([np.zeros(nalphas-1), coefs[:, [1]]])
#    scalings = np.array([0.316, 1.])
#
#    # Impact of s0 on solution
#    s0 = 0.
#    s1, fluxes = rezeq.integrate(delta, s0, alphas, scalings, \
#                a_matrix_noscaling, b_matrix_noscaling)
#    assert fluxes[1]>0
#
#    s0 = alpha_min
#    s1, fluxes = rezeq.integrate(delta, s0, alphas, scalings, \
#                a_matrix_noscaling, b_matrix_noscaling)
#    assert fluxes[1]<0
#
#
#def test_quadrouting(allclose):
#    inflows, outflows = get_data()
#
#    delta = 1
#    q0 = inflows.mean()
#    theta = (inflows.cumsum()-outflows.cumsum()).max()*0.7
#    s0 = 0.
#    sim1 = pd.Series(rezeq.quadrouting(delta, theta, q0, s0, inflows), \
#                        index=inflows.index)
#
#    f = lambda x: -q0*(x/theta)**2
#    nalphas = 500
#    alpha_min = 0
#    alpha_max = 2*theta
#    alphas = np.linspace(alpha_min, alpha_max, nalphas)
#
#    coefs = rezeq.piecewise_linear_approximation(f, alphas)
#    a_matrix_noscaling = np.column_stack([np.ones(nalphas-1), coefs[:, [0]]])
#    b_matrix_noscaling = np.column_stack([np.zeros(nalphas-1), coefs[:, [1]]])
#    scalings = np.column_stack([inflows, np.ones_like(inflows)])
#    s1, fluxes = rezeq.run(delta, s0, alphas, scalings, \
#                            a_matrix_noscaling, b_matrix_noscaling)
#    sim2 = pd.Series(-fluxes[:, 1], index=inflows.index)
#
#    assert allclose(sim1, sim2, atol=5e-2)
#
#
#def test_numrouting(allclose):
#    inflows, outflows = get_data()
#
#    delta = 1
#    q0 = inflows.mean()
#    theta = (inflows.cumsum()-outflows.cumsum()).max()*0.7
#    s0 = 0.
#    sim1 = pd.Series(rezeq.quadrouting(delta, theta, q0, s0, inflows), \
#                        index=inflows.index)
#
#    s2, f2 = rezeq.numrouting(delta, theta, q0, s0, inflows, 2.)
#    sim2 = pd.Series(f2, index=inflows.index)
#
#    assert allclose(sim1, sim2, atol=1e-8)
#
#
#def test_integrate_continuity(allclose):
#    delta = 1
#    nalphas = 10
#    alphas = np.linspace(0, 10, nalphas)
#    theta = 10.
#    u0 = 0.1
#    inflow = 100.
#    q0 = 1.
#
#    f = lambda x: -x**2
#    coefs = rezeq.piecewise_linear_approximation(f, alphas)
#    scalings = np.array([inflow, q0/theta**2])
#    a_matrix_noscaling = np.column_stack([np.ones(nalphas-1), coefs[:, [0]]])
#    b_matrix_noscaling = np.column_stack([np.zeros(nalphas-1), coefs[:, [1]]])
#    b_matrix_noscaling[:, 1] += np.random.uniform(0.2, 1, nalphas-1)
#
#    with pytest.raises(ValueError, match="integrate returns"):
#        u1, fluxes = rezeq.integrate(delta, u0, alphas, scalings, \
#                        a_matrix_noscaling, b_matrix_noscaling)
#
