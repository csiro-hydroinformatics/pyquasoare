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

from pyrezeq import approx, integrate, slow, benchmarks, steady

from test_approx import generate_samples, reservoir_function

from pygme.models import gr4j

import data_reader

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent
LOGGER = iutils.get_logger("integrate", flog=FTEST / "test_integrate.log")

def test_find_alpha(allclose):
    alphas = np.linspace(0, 1, 4)
    for ia, a in enumerate(alphas):
        ialpha = integrate.find_alpha(alphas, a)
        assert ialpha == min(ia, 2)

    u0 = -1.
    ialpha = integrate.find_alpha(alphas, u0)
    assert ialpha == -1

    u0 = 1.1
    ialpha = integrate.find_alpha(alphas, u0)
    assert ialpha == 3

    u0 = 0.2
    ialpha = integrate.find_alpha(alphas, u0)
    assert ialpha == 0

    u0 = 0.4
    ialpha = integrate.find_alpha(alphas, u0)
    assert ialpha == 1

    u0 = 0.7
    ialpha = integrate.find_alpha(alphas, u0)
    assert ialpha == 2


def test_quad_constants(allclose, generate_samples):
    cname, case, params, s0s, Tmax = generate_samples
    ntry = len(params)
    for itry, ((a, b, c), s0) in enumerate(zip(params, s0s)):
        Delta, qD, ssr = integrate.quad_constants(a, b, c)
        assert allclose(Delta, b*b-4*a*c)
        assert allclose(qD, math.sqrt(abs(Delta))/2)
        if a<0 or a>0:
            assert allclose(ssr, b/2/a)


def test_delta_t_max(allclose, generate_samples):
    cname, case, params, s0s, Tmax = generate_samples
    ntry = len(params)
    t_eval = np.logspace(-3, math.log10(Tmax), 1000)

    err_max = 0
    nassessed = 0
    for itry, ((a, b, c), s0) in enumerate(zip(params, s0s)):
        Delta, qD, ssr = integrate.quad_constants(a, b, c)

        # Compute delta max from C code
        delta_tmax = integrate.quad_delta_t_max(a, b, c, Delta, qD, ssr, s0)
        t0 = 0
        tmax = t0+delta_tmax

        # ------- Basic checks ------
        assert delta_tmax>0

        # delta_tmax is infinite for linear function
        # (solution is exponential)
        if approx.isnull(a) or case in [1, 2]:
            assert not np.isfinite(delta_tmax)
            continue

        # Steady state analysis for Delta>=0
        # simulation converges if initial condition is located
        # beyond or below roots depending on sign of a coef
        stdy = steady.quad_steady(a, b, c)
        c1 = a<0 and s0>np.nanmin(stdy)
        c2 = a>0 and s0<np.nanmax(stdy)
        if Delta>=0 and (c1 or c2):
            assert not np.isfinite(delta_tmax)
            continue

        # .. if Delta<0, delta_tmax is always finite
        # because simulation will diverge whatever initial condition is
        if Delta<0:
            assert np.isfinite(delta_tmax)

        # --------- Advanced checks -----------
        # Run solver first to see how far it goes
        f = lambda x: approx.quad_fun(a, b, c, x)
        df = lambda x: approx.quad_grad(a, b, c, x)
        te, ns1, _, _= slow.integrate_forward_numerical(f, df, [], [], t0, s0, t_eval)
        if len(te)==0:
            continue

        if te.max()>Tmax-1e-10:
            assert tmax>=Tmax

        # Compares with slow
        delta_tmax_slow = slow.quad_delta_t_max(a, b, c, Delta, qD, ssr, s0)
        assert np.isclose(delta_tmax, delta_tmax_slow)

        # Check tmax < end of sim
        if te.max()<Tmax and te.max()>0 and len(te)>3:
            # Refines
            t0, t1 = te[-3], te[-1]
            te = np.linspace(t0, 2*t1-t0, 500)
            s0 = ns1[-3]
            te, ns1, nev, njac = slow.integrate_forward_numerical(\
                                                    f, df, [], [], t0, s0, te)
            expected = te.max()

            # Compares with C code
            err = abs(np.log(tmax)-np.log(expected))
            assert err<5e-3
            err_max = max(err, err_max)

        nassessed += 1

    mess = f"[{case}:{cname}] delta tmax: errmax = {err_max:3.2e}"\
            f" assessed={100*nassessed/ntry:0.0f}%"
    LOGGER.info(mess)


def test_forward_vs_finite_difference(allclose, generate_samples):
    cname, case, params, s0s, Tmax = generate_samples
    if case in [7, 10]:
        pytest.skip("Cannot do because numerical round-off")
    ntry = len(params)
    t0 = 0
    errmax_max = 0
    nassessed = 0
    for itry, ((a, b, c), s0) in enumerate(zip(params, s0s)):
        Delta, qD, ssr = integrate.quad_constants(a, b, c)

        # Set integration time
        Tmax = t0+integrate.quad_delta_t_max(a, b, c, Delta, qD, ssr, s0)*0.99
        Tmax = Tmax if np.isfinite(Tmax) else 20.
        t_eval = np.linspace(t0, Tmax, 10000)

        # Integrate with C code
        s1 = integrate.quad_forward(a, b, c, Delta, qD, ssr, t0, s0, t_eval)

        # Compare with slow
        s1_slow = np.array([slow.quad_forward(a, b, c, Delta, qD, ssr, t0, s0, t) \
                                                                    for t in t_eval])
        assert allclose(s1, s1_slow)

        # Test if s1 is monotone
        ds1 = np.abs(np.diff(s1))
        sgn = np.sign(ds1)
        iok = np.abs(ds1)>1e-8
        if iok.sum()>0:
            sgn_ini = np.sign(ds1[iok][0])
            assert np.all(sgn[iok]==sgn_ini)

        # Differentiate using 5th point method
        h = t_eval[1]-t_eval[0]
        ds1 = (-s1[4:]+8*s1[3:-1]-8*s1[1:-3]+s1[:-4])/h/12
        td = t_eval[2:-2]
        # .. compares with expected from ode
        expected = approx.quad_fun(a, b, c, s1[2:-2])
        err = np.abs(np.arcsinh(ds1)-np.arcsinh(expected))

        # Select place where derivative is not too high
        # and after and before the start and end of simul
        ads1 = np.abs(ds1)
        iok = (ads1<1e2) & (td>max(td[3], 1e-3)) \
                        & (td<min(td[-3], td[-1]-1e-3))
        if iok.sum()<4:
            continue

        errmax = np.nanmax(err[iok])
        nassessed += 1
        err_thresh = 1e-6
        assert errmax < err_thresh
        errmax_max = max(errmax, errmax_max)

    LOGGER.info(f"[{case}:{cname}] forward vs finite diff: errmax = {errmax_max:3.2e}"\
                f" assessed={nassessed*100/ntry:0.0f}%")


def test_forward_vs_numerical(allclose, generate_samples):
    cname, case, params, s0s, Tmax = generate_samples
    ntry = len(params)
    t0 = 0
    errmax_max = 0
    nassessed = 0
    for itry, ((a, b, c), s0) in enumerate(zip(params, s0s)):
        Delta, qD, ssr = integrate.quad_constants(a, b, c)

        # Set integration time
        Tmax = min(20, t0+integrate.quad_delta_t_max(a, b, c, Delta, qD, ssr, s0)*0.99)
        if np.isnan(Tmax) or Tmax<0:
            continue
        t_eval = np.linspace(0, Tmax, 1000)

        f = lambda x: approx.quad_fun(a, b, c, x)
        df = lambda x: approx.quad_grad(a, b, c, x)
        te, expected, nev, njac = slow.integrate_forward_numerical(f, df, [], [], \
                                                            t0, s0, t_eval)
        if len(te)<3:
            continue

        dsdt = np.diff(expected)/np.diff(te)
        dsdt = np.insert(dsdt, 0, 0)

        s1 = integrate.quad_forward(a, b, c, Delta, qD, ssr, t0, s0, te)

        err = np.abs(np.arcsinh(s1)-np.arcsinh(expected))
        iok = (np.abs(dsdt)<1e3)
        errmax = np.nanmax(err[iok])
        err_thresh = 5e-3 if case in [5, 7] else 5e-4
        assert errmax<err_thresh
        errmax_max = max(errmax, errmax_max)
        nassessed += 1

    mess = f"[{case}:{cname}] forward vs numerical: "\
                +f"errmax = {errmax_max:3.2e}"\
                +f" assessed={nassessed*100/ntry:0.0f}%"
    LOGGER.info(mess)


def test_inverse(allclose, generate_samples):
    cname, case, params, s0s, Tmax = generate_samples
    ntry = len(params)
    t0 = 0
    nassessed = 0
    errmax_max = 0
    for itry, ((a, b, c), s0) in enumerate(zip(params, s0s)):
        Delta, qD, ssr = integrate.quad_constants(a, b, c)

        # Set integration time
        Tmax = min(20, t0+integrate.quad_delta_t_max(a, b, c, Delta, qD, ssr, s0)*0.9)
        t_eval = np.linspace(0, Tmax, 1000)

        # Simulate
        s1 = integrate.quad_forward(a, b, c, Delta, qD, ssr, t0, s0, t_eval)
        dsdt = approx.quad_fun(a, b, c, s1)

        # Difference with steady state
        stdy = steady.quad_steady(a, b, c)
        stdy[np.isnan(stdy)] = np.inf
        dst1 = np.abs(s1-stdy[0])
        dst2 = np.abs(s1-stdy[1])

        # Compute difference
        ta = integrate.quad_inverse(a, b, c, Delta, qD, ssr, s0, s1)

        iok = (np.abs(s1)<1e20) & (dst1>1e-3) & (dst2>1e-3)
        assert np.all(ta[iok]>=0)

        err = np.abs(np.log(ta)-np.log((t_eval-t0)))
        iok = (np.abs(dsdt)>1e-8) & (dsdt<1e4) & iok
        if iok.sum()<3:
            continue
        errmax = np.nanmax(err[iok])
        err_thresh = 1e-10
        assert errmax< err_thresh
        errmax_max = max(errmax, errmax_max)

        # Compare with slow TODO:fix this!
        iok = np.isfinite(ta)
        #ta_slow = [slow.quad_inverse(a, b, c, Delta, qD, ssr, s0, s) \
        #                        for s in s1[iok]]
        #assert allclose(ta[iok], ta_slow)
        nassessed += 1

    LOGGER.info(f"[{case}:{cname}] inverse: "\
                +f"errmax = {errmax_max:3.2e} "\
                +f"assessed={100*nassessed/ntry:0.0f}%")


def test_increment_fluxes_vs_integration(allclose, generate_samples):
    cname, case, params, nus, s0s, Tmax = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip()

    nskipped = 0
    errbal_max = 0
    errmax_max = 0
    ev = []
    for itry, ((aoj, boj, coj), nu, s0) in enumerate(zip(params, nus, s0s)):
        Delta, qD, ssr = integrate.quad_constants(aoj, boj, coj)
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
        s1 = integrate.quad_forward(aoj, boj, coj, t0, s0, t1)
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

        # Test mass balance
        balance = s1-s0-fluxes.sum()
        errbal_max = max(abs(balance), errbal_max)
        #assert allclose(balance, 0)

        # Compare against numerical integration
        def finteg(t, a, b, c):
            s = integrate.quad_forward(aoj, boj, coj, t0, s0, t)
            return approx.quad_fun(a, b, c, s)

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
        assert allclose(fluxes, fluxes_slow, atol=1e-7)

    mess = f"[{case}:{cname}] fluxes vs integration: "\
                +f"errmax = {errmax_max:3.2e} ({ntry-nskipped} runs) "\
                +f" balmax = {errbal_max:3.3e}"
    LOGGER.info(mess)


def test_reservoir_equation_extrapolation(allclose, ntry, reservoir_function):
    fname, fun, dfun, _, inflow, (alpha0, alpha1) = reservoir_function

    # Reservoir functions
    inp = lambda x: (1.+0*x)*inflow
    sfun = lambda x: inflow+fun(x)
    funs = [inp, fun]

    # Optimize approx
    nalphas = 5
    alphas = np.linspace(alpha0, alpha1, nalphas)
    scr = np.ones(2)
    nu, amat, bmat, cmat, niter, fopt = approx.optimize_nu(funs, alphas, scr)

    # Configure integration
    t0 = 0 # Analytical solution always integrated from t0=0!
    nval = 500
    Tmax = 5
    scalings = np.ones(2)
    t1 = np.linspace(t0, Tmax, nval)
    dalpha = alpha1-alpha0

    for itry in range(-ntry, ntry):
        # Initial conditions outside of approximation range [alpha0, alpha1]
        if itry<0:
            # Initial condition below alpha min
            dd = float(abs(itry+1))/(ntry-1)
            s0 = alpha0-dalpha*dd
        else:
            # Initial condition above alpha max
            dd = float(itry)/(ntry-1)
            s0 = alpha1+dalpha*dd

        # Solve
        sims = [s0]
        s_start = s0
        for i in range(len(t1)-1):
            t_start = t1[i]
            delta = t1[i+1]-t_start
             # integrate - C code
            _, s_end, _ = integrate.integrate(alphas, scalings, nu, \
                                                amat, bmat, cmat, t_start, \
                                                s_start, delta)
            # integrate - python code
            _, s_end_slow, _ = slow.integrate(alphas, scalings, nu, \
                                                amat, bmat, cmat, t_start, \
                                                s_start, delta)
            assert np.isclose(s_end, s_end_slow)
            sims.append(s_end)
            s_start = s_end

        # Expect constant derivative of solution in extrapolation mode
        sims = np.array(sims)
        ilow = sims<alpha0
        dt = t1[1]-t1[0]
        if ilow.sum()>0:
            ds_low = np.diff(sims[ilow])/dt
            expected_low = [approx.quad_fun(amat[0, i], bmat[0, i], \
                                            cmat[0, i], alphas[0]) for i in range(2)]
            expected_low = np.array(expected_low).sum()
            assert allclose(ds_low, expected_low)

        ihigh = sims>alpha1
        if ihigh.sum()>0:
            ds_high = np.diff(sims[ihigh])/dt
            expected_high = [approx.quad_fun(amat[-1, i], bmat[-1, i], \
                                            cmat[-1, i], alphas[-1]) for i in range(2)]
            expected_high = np.array(expected_high).sum()
            assert allclose(ds_high, expected_high)



def test_reservoir_equation(allclose, ntry, reservoir_function):
    fname, fun, dfun, sol, inflow, (alpha0, alpha1) = reservoir_function
    if sol is None:
        pytest.skip("No analytical solution")

    inp = lambda x: (1+0*x)*inflow
    sfun = lambda x: inflow+fun(x)
    funs = [inp, fun]

    dinp = lambda x: 0.
    dsfun = lambda x: dinp(x)+dfun(x)
    dfuns = [dinp, dfun]

    # Optimize approx
    nalphas = 11
    alphas = np.linspace(alpha0, alpha1, nalphas)
    scr = np.ones(2)
    nu, amat, bmat, cmat, niter, fopt = approx.optimize_nu([inp, fun], alphas, scr)
    assert approx.is_continuous(alphas, nu, amat, bmat, cmat)

    # Adjust bounds to avoid numerical problems with analytical solution
    if re.search("^x|^logistic|sin", fname):
        alpha0 += 1e-2
    elif re.search("genlogistic", fname):
        alpha0 += 1e-1
    elif fname == "runge":
        alpha0, alpha1 = -1, 1

    # Configure integration
    scalings = np.ones(2)
    t0 = 0
    nval = 100
    Tmax = 10
    t1 = np.linspace(t0, Tmax, nval)
    errmax_app_max, time_app, niter_app = 0., 0., 0
    errmax_num_max, time_num, niter_num = 0., 0., 0

    for itry in range(ntry):
        s0 = alpha0+(alpha1-alpha0)*float(itry)/(ntry-1)

        # Analytical solution
        expected = sol(t1, s0)

        # Numerical method
        start_exec = time.time()
        tn, fn, nev, njac = slow.integrate_forward_numerical(\
                                        sfun, dsfun, \
                                        funs, dfuns, \
                                        t0, s0, t1)
        end_exec = time.time()
        time_num += (end_exec-start_exec)*1e3
        numerical = fn[:, 0]
        niter_num = max(niter_num, nev+njac)

        # Approximate method
        niter, sims = [0], [s0]
        s_start = s0
        start_exec = time.time()
        for i in range(len(t1)-1):
            t_start = t1[i]
            delta = t1[i+1]-t_start
            # C code
            n, s_end, _ = integrate.integrate(alphas, scalings, nu, \
                                                amat, bmat, cmat, t_start, \
                                                s_start, delta)
            # Python code
            n_slow, s_end_slow, _ = slow.integrate(alphas, scalings, nu, \
                                                amat, bmat, cmat, t_start, \
                                                s_start, delta)
            assert np.isclose(s_end, s_end_slow)

            niter.append(n)
            sims.append(s_end)
            s_start = s_end

        # Process run times
        end_exec = time.time()
        time_app += (end_exec-start_exec)*1e3
        niter = np.array(niter)
        sims = np.array(sims)
        niter_app = max(niter_app, niter.sum())

        # Errors
        isok = ~np.isnan(expected)
        errmax = np.abs(sims[isok]-expected[isok]).max()
        if errmax>errmax_app_max:
            errmax_app_max = errmax
            s0_errmax = s0

        errmax = np.abs(numerical[isok]-expected[isok]).max()
        errmax_num_max = max(errmax, errmax_num_max)

    err_thresh = {
        "x2": 1e-9, \
        "x4": 1e-3, \
        "x6": 1e-3, \
        "x8": 1e-3, \
        "tanh": 1e-2, \
        "exp": 1e-7, \
        "sin": 5e-3, \
        "runge": 1e-3, \
        "stiff": 1e-9, \
        "ratio": 5e-2, \
        "logistic": 1e-7, \
        "genlogistic": 5e-2
    }
    assert errmax_app_max < err_thresh[fname]
    assert time_app<time_num*0.95

    LOGGER.info(f"[{fname}] approx vs analytical: errmax={errmax_app_max:3.2e}"\
                    +f" / time={time_app:3.3e}ms / niter={niter_app}")

    LOGGER.info(f"[{fname}] numerical vs analytical: "\
                    +f"errmax = {errmax_num_max:3.2e} "\
                    +f"(ratio vs app={errmax_app_max/errmax_num_max:0.2f})"\
                    +f" / time={time_num:3.3e}ms "\
                    +f"(ratio vs app={time_app/time_num:0.2f})"\
                    +f" / niter={niter_num}")



def test_reservoir_equation_gr4j(allclose):
    nalphas = 20
    alphas = np.linspace(0., 1.2, nalphas)
    start, end = "2017-01", "2022-12-31"
    nsubdiv = 10000
    nu = 0.01
    X1s = [50, 200, 1000]
    LOGGER.info("")

    # Compute approx coefs
    fluxes, _ = benchmarks.gr4jprod_fluxes_noscaling()
    amat = np.zeros((nalphas-1, 3))
    bmat = amat.copy()
    cmat = amat.copy()
    for j in range(nalphas-1):
        a0, a1 = alphas[[j, j+1]]
        for i in range(3):
            a, b, c, _ = approx.get_coefficients(fluxes[i], a0, a1, nu)
            amat[j, i] = a
            bmat[j, i] = b
            cmat[j, i] = c

    # Loop over sites
    for isite, siteid in enumerate(data_reader.SITEIDS):
        # Get climate data
        df = data_reader.get_data(siteid, "daily")
        df = df.loc[start:end]
        nval = len(df)
        inputs = np.ascontiguousarray(df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]])

        for X1 in X1s:
            # Run approximate solution
            s0 = X1/2
            expected = benchmarks.gr4jprod(nsubdiv, X1, s0, inputs)
            scalings = np.ones(3)
            s_start = s0/X1

            # Run solution based on approximation function
            sims = np.zeros_like(expected)
            for t in range(nval):
                P, E = inputs[t]

                # Apply interception to inputs
                pi = max(0, P-E)/X1
                ei = max(0, E-P)/X1
                scalings[:2] = pi, ei

                # Integate equation
                n, s_end, fx = integrate.integrate(alphas, scalings, nu, \
                                                    amat, bmat, cmat, 0., \
                                                    s_start, 1)
                sims[t, 0] = s_end*X1
                sims[t, 1] = fx[0]*X1
                sims[t, 2] = -fx[1]*X1
                sims[t, 3] = -fx[2]*X1
                s_start = s_end

            errmax = np.abs(sims-expected).max()
            mess = f"approx vs gr4jprod /site {isite+1}/x1={X1:4.0f}:"\
                        +f" errmax={errmax:3.3e}"
            LOGGER.info(mess)
            atol = 1e-3
            rtol = 1e-4
            assert np.allclose(expected, sims, atol=atol, rtol=rtol)

