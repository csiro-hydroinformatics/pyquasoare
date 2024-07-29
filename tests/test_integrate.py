from pathlib import Path
from itertools import product as prod
import time
import math
import re
import pytest

import numpy as np
import pandas as pd
import scipy.integrate as sci_integrate

from hydrodiy.io import iutils

from pyrezeq import approx, integrate, slow

from test_approx import generate_samples, reservoir_function

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent

LOGGER = iutils.get_logger("integrate", flog=FTEST / "test_integrate.log")


# ----- UTILITY FUNCTIONS ------------------------------------------
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


# ----- TESTS ------------------------------------------

def test_find_alphas(allclose):
    alphas = np.linspace(0, 1, 4)
    u0 = -1.
    ialpha = integrate.find_alpha(alphas, u0)
    assert ialpha == 0

    u0 = 1.1
    ialpha = integrate.find_alpha(alphas, u0)
    assert ialpha == 2

    u0 = 0.2
    ialpha = integrate.find_alpha(alphas, u0)
    assert ialpha == 0

    u0 = 0.4
    ialpha = integrate.find_alpha(alphas, u0)
    assert ialpha == 1

    u0 = 0.7
    ialpha = integrate.find_alpha(alphas, u0)
    assert ialpha == 2



def test_delta_t_max(allclose, generate_samples, printout):
    cname, case, params, nus, s0s, Tmax = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("Skip param config")

    LOGGER.info("")
    nprint = 50
    t_eval = np.linspace(0, Tmax, 1000)

    err_max = 0
    nskipped = 0
    ndpos = 0
    ndelta = 0
    for itry, ((a, b, c), nu, s0) in enumerate(zip(params, nus, s0s)):

        # Log progress
        if itry%nprint==0 and printout:
            LOGGER.info(f"delta tmax - case {case} - Try {itry+1:4d}/{ntry:4d}")

        # Run solver first to see how far it goes
        t0 = 0
        f = lambda x: approx.approx_fun(nu, a, b, c, x)
        df = lambda x: approx.approx_jac(nu, a, b, c, x)
        te, ns1, nev, njac = slow.integrate_forward_numerical(\
                                    [f], [df], t0, [s0], t_eval)

        # Check tmax < end of sim
        if te.max()<Tmax and te.max()>0 and len(te)>3:
            Delta = a**2-4*b*c
            ndpos += Delta>0

            # Refines
            t0, t1 = te[[-3, -1]]
            te = np.linspace(t0, 2*t1-t0, 1000)
            s0 = ns1[-3]
            te, ns1, nev, njac = slow.integrate_forward_numerical(\
                                                    [f], [df], t0, [s0], te)
            expected = te.max()

            s1 = integrate.integrate_forward(nu, a, b, c, t0, s0, te)
            dtm = integrate.delta_t_max(nu, a, b, c, s0)
            assert dtm>0
            tm = t0+dtm

            err = abs(np.log(tm)-np.log(expected))
            assert err<2e-3
            #plot_solution(te, s1, ns1, show=True, params=[nu, a, b, c])
            err_max = max(err, err_max)

            dtm_slow = slow.integrate_delta_t_max(nu, a, b, c, s0)
            assert np.isclose(dtm, dtm_slow)

        else:
            nskipped += 1

    mess = f"delta tmax - Case {cname}: errmax = {err_max:3.2e}"\
            f"  skipped={100*nskipped/ntry:0.0f}%"
    if case>=9:
        if nskipped<ntry:
            mess += f" - Delta>0={100*ndpos/(ntry-nskipped):0.0f}%"

    LOGGER.info(mess)


def test_forward_vs_finite_difference(allclose, generate_samples, printout):
    cname, case, params, nus, s0s, Tmax = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()

    LOGGER.info("")
    nprint = 50
    t0 = 0
    errmax_max = 0
    notskipped = 0
    ndelta = 0
    for itry, ((a, b, c), nu, s0) in enumerate(zip(params, nus, s0s)):
        # Log progress
        if itry%nprint==0 and printout:
            LOGGER.info(f"forward vs finite diff - case {case} - Try {itry+1:4d}/{ntry:4d}")

        # Set integration time
        Tmax = min(20, t0+integrate.delta_t_max(nu, a, b, c, s0)*0.99)
        if np.isnan(Tmax) or Tmax<0:
            continue
        t_eval = np.linspace(0, Tmax, 1000)

        s1 = integrate.integrate_forward(nu, a, b, c, t0, s0, t_eval)
        if np.all(np.isnan(s1)):
            continue

        Tmax_rev = t_eval[~np.isnan(s1)].max()
        if Tmax_rev<1e-10:
            continue

        t_eval_rev = np.linspace(0, Tmax_rev, 10000)
        s1 = integrate.integrate_forward(nu, a, b, c, t0, s0, t_eval_rev)

        # Compare with slow
        s1_slow = [slow.integrate_forward(nu, a, b, c, t0, s0, t) \
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

        expected = approx.approx_fun(nu, a, b, c, s1[2:-2])

        err = np.abs(np.arcsinh(ds1*1e-3)-np.arcsinh(expected*1e-3))
        iok = (np.abs(ds1)<1e2) & (td>td[2]) & (td<td[-2])
        if iok.sum()<4:
            continue

        errmax = np.nanmax(err[iok])
        notskipped += 1
        assert errmax<1e-3
        errmax_max = max(errmax, errmax_max)

    LOGGER.info(f"forward vs finite diff - Case {cname}: errmax = {errmax_max:3.2e}"\
                f" skipped={(ntry-notskipped)*100/ntry:0.0f}%")


def test_forward_vs_numerical(allclose, generate_samples, printout):
    cname, case, params, nus, s0s, Tmax = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()

    LOGGER.info("")
    nprint = 50
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
            LOGGER.info(f"forward vs numerical - {cname} - Try {itry+1:4d}/{ntry:4d}")

        # Set integration time
        Tmax = min(20, t0+integrate.delta_t_max(nu, a, b, c, s0)*0.99)
        if np.isnan(Tmax) or Tmax<0:
            continue
        t_eval = np.linspace(0, Tmax, 1000)

        f = lambda x: approx.approx_fun(nu, a, b, c, x)
        df = lambda x: approx.approx_jac(nu, a, b, c, x)
        te, expected, nev, njac = slow.integrate_forward_numerical([f], [df], \
                                                            t0, [s0], t_eval)
        if len(te)<3:
            nskipped += 1
            continue

        dsdt = np.diff(expected)/np.diff(te)
        dsdt = np.insert(dsdt, 0, 0)

        s1 = integrate.integrate_forward(nu, a, b, c, t0, s0, te)

        err = np.abs(np.arcsinh(s1*1e-3)-np.arcsinh(expected*1e-3))
        iok = np.abs(dsdt)<1e3
        errmax = np.nanmax(err[iok])
        assert errmax<1e-4
        errmax_max = max(errmax, errmax_max)

    perc_skipped = nskipped*100/ntry
    mess = f"forward vs numerical - Case {cname}: errmax = {errmax_max:3.2e}  Skipped={perc_skipped:0.0f}%"
    if case>=8:
        perc_delta_pos = perc_delta_pos/ndelta*100
        mess += f"  Delta>0 = {perc_delta_pos:0.0f}%"
    LOGGER.info(mess)


def test_inverse(allclose, generate_samples, printout):
    cname, case, params, nus, s0s, Tmax = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()

    LOGGER.info("")
    nprint = 50
    t0 = 0
    nskipped = 0
    errmax_max = 0
    for itry, ((a, b, c), nu, s0) in enumerate(zip(params, nus, s0s)):
        if itry%nprint==0 and printout:
            LOGGER.info(f"inverse - Case {cname} - Try {itry+1:4d}/{ntry:4d}")

        # Set integration time
        Tmax = min(20, t0+integrate.delta_t_max(nu, a, b, c, s0)*0.99)
        if np.isnan(Tmax) or Tmax<0:
            nskipped += 1
            continue
        t_eval = np.linspace(0, Tmax, 1000)

        # Simulate
        s1 = integrate.integrate_forward(nu, a, b, c, t0, s0, t_eval)
        ds1 = approx.approx_fun(nu, a, b, c, s1)

        iok = ~np.isnan(s1) & (ds1>1e-5)
        iok[0] = False
        if iok.sum()<5:
            nskipped += 1
            continue

        t, s1 = t_eval[iok], s1[iok]

        # Compute difference
        ta = integrate.integrate_inverse(nu, a, b, c, s0, s1)
        assert np.all(ta>=0)

        dsdt = approx.approx_fun(nu, a, b, c, s1)
        err = np.abs(np.log(ta*1e-3)-np.log((t-t0)*1e-3))
        iok = (dsdt>1e-4) & (dsdt<1e4)
        if iok.sum()<5:
            nskipped += 1
            continue

        errmax = np.nanmax(err[iok])
        assert errmax< 5e-6 if case in [9, 12] else 1e-8
        errmax_max = max(errmax, errmax_max)

        # Compare with slow
        ta_slow = [slow.integrate_inverse(nu, a, b, c, s0, s) for s in s1]
        assert allclose(ta, ta_slow)

    perc_skipped = nskipped*100/ntry
    mess = f"inverse - Case {cname}: errmax = {errmax_max:3.2e}  Skipped={perc_skipped:0.0f}%"
    LOGGER.info(mess)


def test_increment_fluxes_vs_integration(allclose, \
                                        generate_samples, printout):
    cname, case, params, nus, s0s, Tmax = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()

    LOGGER.info("")
    nprint = 50
    nskipped = 0
    errbal_max = 0
    errmax_max = 0
    ev = []
    for itry, ((aoj, boj, coj), nu, s0) in enumerate(zip(params, nus, s0s)):
        if itry%nprint==0 and printout:
            LOGGER.info(f"fluxes vs integration - case {case} - Try {itry+1:4d}/{ntry:4d}")

        avect, bvect, cvect = np.random.uniform(-1, 1, size=(3, 3))

        # make sure coefficient sum matches aoj, boj and coj
        sa, sb, sc = avect.sum(), bvect.sum(), cvect.sum()
        avect += (aoj-sa)/3
        bvect += (boj-sb)/3
        cvect += (coj-sc)/3

        # Integrate forward analytically
        t1 = min(10, integrate.delta_t_max(nu, aoj, boj, coj, s0))
        t0 = t1*0.05 # do not start at zero to avoid sharp falls
        t1 = t1*0.5 # far away from limits of validity
        s1 = integrate.integrate_forward(nu, aoj, boj, coj, t0, s0, t1)
        if np.isnan(s1):
            nskipped += 1
            continue

        # Check error if sum of coefs is not matched
        with pytest.raises(ValueError):
            cvect2 = cvect.copy()
            cvect2[0] += 10
            fluxes = np.zeros(3)
            integrate.increment_fluxes(nu, avect, bvect, cvect2, \
                            aoj, boj, coj, t0, t1, s0, s1, fluxes)

        # Compute fluxes analytically
        fluxes = np.zeros(3)
        integrate.increment_fluxes(nu, avect, bvect, cvect, \
                        aoj, boj, coj, t0, t1, s0, s1, fluxes)

        a, b, c = aoj, boj, coj
        Delta = a**2-4*b*c

        # Test mass balance
        balance = s1-s0-fluxes.sum()
        errbal_max = max(abs(balance), errbal_max)
        #assert allclose(balance, 0)

        # Compare against numerical integration
        def finteg(t, a, b, c):
            s = integrate.integrate_forward(nu, aoj, boj, coj, t0, s0, t)
            return approx.approx_fun(nu, a, b, c, s)

        expected = np.array([sci_integrate.quad(finteg, t0, t1, args=(a, b, c))\
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
        slow.increment_fluxes(nu, avect, bvect, cvect, \
                        aoj, boj, coj, t0, t1, s0, s1, fluxes_slow)
        assert allclose(fluxes, fluxes_slow)

    mess = f"fluxes vs integration - Case {cname}: errmax = {errmax_max:3.2e} ({ntry-nskipped} runs) "+\
                f" Balmax = {errbal_max:3.3e}"
    LOGGER.info(mess)


def test_reservoir_equation_extrapolation(allclose, reservoir_function):
    fname, fun, dfun, _, inflow, (alpha0, alpha1) = reservoir_function

    # Reservoir functions
    inp = lambda x: (1.+0*x)*inflow
    sfun = lambda x: inflow+fun(x)
    funs = [inp, fun]

    # Optimize approx
    nalphas = 5
    alphas = np.linspace(alpha0, alpha1, nalphas)
    nu, amat, bmat, cmat = approx.optimize_nu(funs, alphas)

    # Configure integration
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
        _, s_end_low, _ = integrate.integrate(alphas, scalings, nu, \
                                            amat, bmat, cmat, t_start, \
                                            s_start_low, delta)
        # .. compare against slow
        _, s_end_low_slow, _ = slow.integrate(alphas, scalings, nu, \
                                            amat, bmat, cmat, t_start, \
                                            s_start_low, delta)
        assert np.isclose(s_end_low, s_end_low_slow)
        approx_low.append(s_end_low)
        s_start_low = s_end_low

        _, s_end_high, _ = integrate.integrate(alphas, scalings, nu, \
                                            amat, bmat, cmat, t_start, \
                                            s_start_high, delta)
        # .. compare against slow
        _, s_end_high_slow, _ = slow.integrate(alphas, scalings, nu, \
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
    expected_low = [approx.approx_fun(nus[0], amat[0, i], bmat[0, i], \
                    cmat[0, i], alphas[0]) for i in range(2)]
    expected_low = np.array(expected_low).sum()
    assert allclose(ds_low, expected_low)

    approx_high = np.array(approx_high)
    ihigh = approx_high>alpha1
    assert ihigh.sum()>0
    ds_high = np.diff(approx_high[ihigh])/dt
    expected_high = [approx.approx_fun(nus[-1], amat[-1, i], bmat[-1, i], \
                    cmat[-1, i], alphas[-1]) for i in range(2)]
    expected_high = np.array(expected_high).sum()
    assert allclose(ds_high, expected_high)



def test_reservoir_equation(allclose, ntry, reservoir_function):
    fname, fun, dfun, sol, inflow, (alpha0, alpha1) = reservoir_function
    if sol is None:
        pytest.skip("No analytical solution")

    LOGGER.info("")

    inp = lambda x: (1+0*x)*inflow
    sfun = lambda x: inflow+fun(x)
    funs = [sfun, inp, fun]

    dinp = lambda x: 0.
    dsfun = lambda x: dinp(x)+dfun(x)
    dfuns = [dsfun, dinp, dfun]

    # Optimize approx
    nalphas = 11
    alphas = np.linspace(alpha0, alpha1, nalphas)
    nu, amat, bmat, cmat = approx.optimize_nu([inp, fun], alphas)

    # Configure integration
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
        niter, sims = [0], [s0]
        s_start = s0
        start_exec = time.time()
        for i in range(len(t1)-1):
            t_start = t1[i]
            #LOGGER.info(f"\n### [{i:3d}] t_start = {t_start} / s_start = {s_start:0.5f}###")
            delta = t1[i+1]-t_start
            n, s_end, _ = integrate.integrate(alphas, scalings, nu, \
                                                amat, bmat, cmat, t_start, \
                                                s_start, delta)
            # Against slow
            #n_slow, s_end_slow, _ = slow.integrate(alphas, scalings, nus, \
            #                                    amat, bmat, cmat, t_start, \
            #                                    s_start, delta)
            #assert np.isclose(s_end, s_end_slow)
            niter.append(n)
            sims.append(s_end)
            s_start = s_end

        #LOGGER.info("\n######### Finito ##########")

        end_exec = time.time()
        time_app += (end_exec-start_exec)*1e3
        niter = np.array(niter)
        sims = np.array(sims)
        niter_app = max(niter_app, niter.sum())

        # Numerical method
        start_exec = time.time()
        tn, fn, nev, njac = slow.integrate_forward_numerical(\
                                        funs, dfuns, \
                                        t0, [s0]+[0]*2, t1)
        end_exec = time.time()
        time_num += (end_exec-start_exec)*1e3
        numerical = fn[:, 0]
        niter_num = max(niter_num, nev+njac)

        # Errors
        errmax = np.abs(sims-expected).max()
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
        sims = [s0]
        for i in range(len(t1)-1):
            start = t1[i]
            delta = t1[i+1]-start
            n, s_end, _ = integrate.integrate(alphas, scalings, nus, \
                                                amat, bmat, cmat, start, \
                                                s_start, delta)
            sims.append(s_end)
            s_start = s_end

        sims = np.array(sims)

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

        ds = approx.approx_fun_from_matrix(alphas, nus, \
                                        amat, bmat, cmat, xx)
        ds = ds.sum(axis=1)
        ax.plot(ds, xx, label="approx", color="tab:orange")

        ds = approx.approx_fun_from_matrix(alphas, nus, \
                                        amat, bmat, cmat, approx)
        ds = ds.sum(axis=1)
        ax.plot(ds, sims, lw=4, label="expected sim", color="tab:orange")
        ax.plot(ds[0], sims[0], "x", ms=5, color="tab:orange")
        putils.line(ax, 0, 1, 0, 0, "k-", lw=0.8)

        #for j in range(nalphas-1):
        #    nu, a, b, c = nus[j], amat.sum(axis=1)[j], bmat.sum(axis=1)[j], cmat.sum(axis=1)[j]
        #    ds = approx.approx_fun(nu, a, b, c, xx)
        #    ax.plot(ds, xx, label=f"approx {j}")

        ax.legend()

        ax = axs[1]
        ax.plot(t1, expected, label="expected")
        ax.plot(t1, approx, label="approx")
        ax.legend()
        ax.set(title=f"fun={fname} s0={s0_errmax:0.5f}")
        plt.show()
        import pdb; pdb.set_trace()


    LOGGER.info(f"approx vs analytical {fname}: errmax={errmax_app_max:3.2e}"\
                    +f" / time={time_app:3.3e}ms / niter={niter_app}")
    LOGGER.info(f"numerical vs analytical {fname}: errmax = {errmax_num_max:3.2e}"\
                    +f" / time={time_num:3.3e}ms / niter={niter_num}")
