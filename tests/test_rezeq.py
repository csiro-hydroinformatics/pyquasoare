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
from pyrezeq import rezeq, rezeq_slow

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent

NCASES = 12
PARAM_MAX = 5
S0_MAX = 5
NU_MAX = 5

# ----- FIXTURES ---------------------------------------------------
@pytest.fixture(scope="module", \
            params=["x2", "x4", "x6", "tanh", "exp", "sin", "recip", "recipquad"])
def reservoir_function(request):
    name = request.param
    # sol is the analytical solution of ds/dt = inflow+fun(s)
    # except for
    if name == "x2":
        sol = lambda t, s0: (s0+np.tanh(t))/(1+s0*np.tanh(t))
        return name, lambda x: -x**2, lambda x: -2*x, sol, 1.

    elif name == "x4":
        return name, lambda x: -x**4, lambda x: -4*x**3, None, None

    elif name == "x6":
        return name, lambda x: -x**6, lambda x: -6*x**5, None, None

    elif name == "tanh":
        a, b = 0.5, 10
        sol = lambda t, s0: (np.asinh(2*np.exp(-t/b)+np.sinh(a+b*s0))-a)/b
        return name, lambda x: -np.tanh(a+b*x), \
                            lambda x: b*(np.tanh(a+b*x)**2-1), sol, 0.

    elif name == "exp":
        sol = lambda t, s0: s0+t-np.log(1-(1+np.exp(t))/math.exp(s0))
        return name, lambda x: -np.exp(x), lambda x: -np.exp(x), sol, 1.

    elif name == "sin":
        w = 2*math.pi
        sol = lambda t, s0: -2./w*np.atan(np.exp(w*t)+np.tan(w*s0/2))
        return name, lambda x: np.sin(w*x), lambda x: -w*np.cos(w*x), sol, 0.

    elif name == "recip":
        return name, lambda x: -1e-2/(1.01-x), lambda x: 1e-2/(1.01-x)**2, None, None

    elif name == "recipquad":
        return name, lambda x: -1e-4/(1.01-x)**2, lambda x: 2e-4/(1.01-x)**3, None, None


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
    elif case ==11:
        name = "General case with large scaling"
        params *= 1000
    elif case ==12:
        name = "General case with low scaling"
        params /= 1000

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
    return nus, s0s, Tmax

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


def test_integrate_delta_t_max(allclose, parameter_samples, printout):
    case, params, cname = parameter_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("Skip param config")
    nprint = get_nprint(ntry)
    nus, s0s, Tmax = sample_config(ntry)
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
        te, ns1 = rezeq_slow.integrate_forward_numerical([f], [df], t0, [s0], t_eval)

        # Check tmax < end of sim
        if te.max()<Tmax and te.max()>0 and len(te)>3:
            Delta = a**2-4*b*c
            ndpos += Delta>0

            # Refines
            t0, t1 = te[[-3, -1]]
            te = np.linspace(t0, 2*t1-t0, 1000)
            s0 = ns1[-3]
            te, ns1 = rezeq_slow.integrate_forward_numerical([f], [df], t0, [s0], te)
            expected = te.max()

            s1 = rezeq.integrate_forward(nu, a, b, c, t0, s0, te)
            dtm = rezeq.integrate_delta_t_max(nu, a, b, c, s0)
            assert dtm>0
            tm = t0+dtm

            err = abs(np.log(tm)-np.log(expected))
            assert err<2e-3
            err_max = max(err, err_max)
        else:
            nskipped += 1

    mess = " "*4+f">> Errmax = {err_max:3.2e}"\
            f"  skipped={100*nskipped/ntry:0.0f}%"
    if case >=8:
        if nskipped<ntry:
            mess += f" - when running we have Delta>0={100*ndpos/(ntry-nskipped):0.0f}%"

    print(mess)
    print("")


def test_steady_state(allclose, parameter_samples):
    case, params, cname = parameter_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("Skip param config")
    nprint = get_nprint(ntry)

    print("")
    print(" "*4+f"Testing steady state - case {case} / {cname}")
    err_max = 0
    a, b, c = [np.ascontiguousarray(v) for v in params.T]
    steady = rezeq.steady_state(nus, a, b, c)

    if case<5:
        # No steady state
        assert np.all(np.isnan(steady))
        print(" "*4 +">> No steady state for this case")
        return

    # check nan values
    iboth = np.isnan(steady).sum(axis=1)==0
    assert np.all(np.diff(steady[iboth], axis=1)>=0)

    if case==10:
        # 2 distinct roots
        assert np.all(np.diff(steady[iboth], axis=1)>0)

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


def test_integrate_forward_vs_finite_difference(allclose, parameter_samples, printout):
    case, params, cname = parameter_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()
    nprint = get_nprint(ntry)
    nus, s0s, _ = sample_config(ntry)

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
        Tmax = min(20, t0+rezeq.integrate_delta_t_max(nu, a, b, c, s0)-1e-2)
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


def test_integrate_forward_vs_numerical(allclose, parameter_samples, printout):
    case, params, cname = parameter_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()
    nprint = get_nprint(ntry)
    nus, s0s, _ = sample_config(ntry)

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
        Tmax = min(20, t0+rezeq.integrate_delta_t_max(nu, a, b, c, s0)-1e-2)
        if np.isnan(Tmax) or Tmax<0:
            continue
        t_eval = np.linspace(0, Tmax, 1000)

        f = lambda x: rezeq.approx_fun(nu, a, b, c, x)
        df = lambda x: rezeq.approx_jac(nu, a, b, c, x)
        te, expected = rezeq_slow.integrate_forward_numerical([f], [df], \
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


def test_integrate_inverse(allclose, parameter_samples, printout):
    case, params, cname = parameter_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()
    nprint = get_nprint(ntry)
    nus, s0s, _ = sample_config(ntry)

    print("")
    print(" "*4+f"Testing integrate_inverse - case {case} / {cname}")
    t0 = 0
    nskipped = 0
    errmax_max = 0
    for itry, ((a, b, c), nu, s0) in enumerate(zip(params, nus, s0s)):
        if itry%nprint==0 and printout:
            print(" "*8+f"inverse - case {case} - Try {itry+1:4d}/{ntry:4d}")

        # Set integration time
        Tmax = min(20, t0+rezeq.integrate_delta_t_max(nu, a, b, c, s0)-1e-2)
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

    perc_skipped = nskipped*100/ntry
    mess = f"Errmax = {errmax_max:3.2e}  Skipped={perc_skipped:0.0f}%"
    print(" "*4+mess)
    print("")


def test_get_coefficients(allclose, reservoir_function):
    # Get function and its derivative
    fname, fun, dfun, _, _ = reservoir_function
    alphaj = 0.
    alphajp1 = 1.
    nus = [0.01, 0.1, 1, 5]

    for eps, nu in prod([None, 0.2, 0.5, 0.8], nus):
        if eps==-1 and nu>nus[0]:
            continue
        a, b, c = rezeq.get_coefficients(fun, alphaj, alphajp1, nu, eps)

        # Check continuity
        assert allclose(rezeq.approx_fun(nu, a, b, c, alphaj), fun(alphaj))
        assert allclose(rezeq.approx_fun(nu, a, b, c, alphajp1), fun(alphajp1))

        # Check mid-point values
        if eps is None:
            continue
        elif eps==-1:
            assert b==-c
        else:
            x = (1-eps)*alphaj+eps*alphajp1
            assert allclose(rezeq.approx_fun(nu, a, b, c, x), fun(x))



def test_get_coefficients_matrix(allclose, selfun, reservoir_function):
    # Get function and its derivative
    fname, fun, dfun, _, _ = reservoir_function
    if selfun !="":
        if selfun != fname:
            pytest.skip("Skip non selected function")

    funs = [lambda x: 1., fun]
    nalphas = 500
    alphas = np.linspace(0., 1., nalphas)
    nus = [0.01, 1, 2, 5, 8]
    s = np.linspace(-0.1, 1.1, 1000)
    print("")
    errmax = {n: np.inf for n in nus}
    #for e, nu in prod([-1, 0, 0.2, 0.5, 0.8, 1], nus):
    for e, nu in prod([None], nus):
        if e==-1 and nu>nus[0]:
            continue

        n = nu*np.ones(nalphas-1)
        eps = None if e is None else e*np.ones(nalphas-1)
        _, amat, bmat, cmat = rezeq.get_coefficients_matrix(funs, \
                                                        alphas, n, eps)
        # Run approx
        out = rezeq.approx_fun_from_matrix(alphas, n, amat, bmat, cmat, s)
        fapprox = out[:, 1]
        assert allclose(fapprox[s<0], fun(0))
        assert allclose(fapprox[s>1], fun(1))

        ftrue = fun(s)
        err = fapprox-ftrue
        isin = (s>=0)&(s<=1)
        emax = np.abs(err[isin]).max()
        errmax[nu] = min(errmax[nu], emax)

        if fname in "sin":
            ethresh = 1e-6
        elif fname == "recip":
            ethresh = 1e-4
        elif fname == "recipquad":
            ethresh = 1e-3
        else:
            ethresh = 1e-7
        assert emax < ethresh

    print(f"coef matrix - fun={fname} :")
    for nu in nus:
        print(" "*4+f"errmax(nu={nu:0.2f}) = {errmax[nu]:3.3e}")


def test_steady_state_scalings(allclose):
    nalphas = 500
    alphas = np.linspace(0, 1.2, nalphas)
    # GR4J production
    funs = [
        lambda x: 1-x**2, \
        lambda x: -x*(2-x), \
        lambda x: -x**4/3
    ]
    nus, amat, bmat, cmat = rezeq.get_coefficients_matrix(funs, alphas)

    nval = 200
    scalings = np.random.uniform(0, 100, size=(nval, 3))
    scalings[:, -1] = 1

    steady = rezeq.steady_state_scalings(alphas, nus, scalings, amat, bmat, cmat)
    for t in range(nval):
        s0 = steady[t]
        # Check steady on approx fun
        amats = amat*scalings[t][None, :]
        bmats = bmat*scalings[t][None, :]
        cmats = cmat*scalings[t][None, :]
        out = rezeq.approx_fun_from_matrix(alphas, nus, amats, bmats, cmats, s0)
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


def test_increment_fluxes(allclose, parameter_samples, printout):
    case, params, cname = parameter_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()
    nprint = get_nprint(ntry)
    nus, s0s, _ = sample_config(ntry)

    print("")
    print(" "*4+f"Testing increment_fluxes - case {case} / {cname}")
    t0 = 0
    nskipped = 0
    errmax_max = 0
    ntry = ntry//3
    for itry in range(ntry):
        if itry%nprint==0 and printout:
            print(" "*8+f"flux - case {case} - Try {itry+1:4d}/{ntry:4d}")
        nu, s0 = nus[itry], s0s[itry]
        avect = params[3*itry:3*itry+3, 0]
        bvect = params[3*itry:3*itry+3, 1]
        cvect = params[3*itry:3*itry+3, 2]
        aoj = avect.sum()
        boj = bvect.sum()
        coj = cvect.sum()
        fluxes = np.zeros(3)
        t0, t1 = 0, 10

        # Integrate numerical
        sfun = lambda x: rezeq.approx_fun(nu, aoj, boj, coj, x)
        funs = [sfun] + [lambda x: rezeq.approx_fun(nu, a, b, c, x) \
                    for a, b, c in zip(avect, bvect, cvect)]

        sdfun = lambda x: rezeq.approx_jac(nu, aoj, boj, coj, x)
        dfuns = [sdfun] + [lambda x: rezeq.approx_jac(nu, a, b, c, x) \
                    for a, b, c in zip(avect, bvect, cvect)]

        #
        import pdb; pdb.set_trace()
        s1 = rezeq.integrate_forward(nu, aoj, boj, coj, s0, t0, t1)


    rezeq.increment_fluxes(nus, scalings, \
                        avect, bvect, cvect, \
                        aoj, boj, coj, \
                        t0, t1, s0, s1, fluxes)


def test_integrate_reservoir_equations(allclose, selfun, reservoir_function):
    fname, fun, dfun, sol, inflow = reservoir_function
    if selfun !="":
        if selfun != fname:
            pytest.skip("Skip non selected function")

    if sol is None:
        pytest.skip("No analytical solution")

    inp = lambda x: inflow
    sfun = lambda x: inflow+fun(x)
    funs = [sfun, inp, fun]

    dinp = lambda x: 0.
    dsfun = lambda x: dinp(x)+dfun(x)
    dfuns = [dsfun, dinp, dfun]

    nalphas = 500
    alphas = np.linspace(0., 1., nalphas)
    nus, alphase, amat, bmat, cmat = rezeq.get_coefficients_matrix(funs, \
                                                                alphas, nus=1)
    s0 = [-5, 0, 0]
    t0, Tmax = 0, 10
    t_eval = np.linspace(t0, Tmax, 500)
    te, expected = rezeq_slow.integrate_forward_numerical(funs, dfuns, t0, s0, t_eval)

    import pdb; pdb.set_trace()

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
