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
from pyrezeq import rezeq, rezeq_slow, optimize

from test_rezeq import reservoir_function

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent


def test_integrate_reservoir_equation_extrapolation(allclose, reservoir_function):
    fname, fun, dfun, _, inflow, (alpha0, alpha1) = reservoir_function
    print("")
    print(" "*4+f"Testing rezeq integrate extrapolation - fun {fname} ")

    # Reservoir functions
    inp = lambda x: (1.+0*x)*inflow
    sfun = lambda x: inflow+fun(x)
    funs = [inp, fun]

    # Optimize approx
    nalphas = 5
    alphas = np.linspace(alpha0, alpha1, nalphas)
    nu, epsilon, amat, bmat, cmat = optimize.optimize_nu_and_epsilon(funs, alphas)

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
        _, s_end_low, _ = rezeq.integrate(alphas, scalings, nu, \
                                            amat, bmat, cmat, t_start, \
                                            s_start_low, delta)
        # .. compare against slow
        _, s_end_low_slow, _ = rezeq_slow.integrate(alphas, scalings, nu, \
                                            amat, bmat, cmat, t_start, \
                                            s_start_low, delta)
        assert np.isclose(s_end_low, s_end_low_slow)
        approx_low.append(s_end_low)
        s_start_low = s_end_low

        _, s_end_high, _ = rezeq.integrate(alphas, scalings, nu, \
                                            amat, bmat, cmat, t_start, \
                                            s_start_high, delta)
        # .. compare against slow
        _, s_end_high_slow, _ = rezeq_slow.integrate(alphas, scalings, nu, \
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
    inp = lambda x: (1+0*x)*inflow
    sfun = lambda x: inflow+fun(x)
    funs = [sfun, inp, fun]

    dinp = lambda x: 0.
    dsfun = lambda x: dinp(x)+dfun(x)
    dfuns = [dsfun, dinp, dfun]

    # Optimize approx
    nalphas = 11
    alphas = np.linspace(alpha0, alpha1, nalphas)
    nu, epsilon, amat, bmat, cmat = optimize.optimize_nu_and_epsilon([inp, fun], alphas)

    print("\n"+" "*4+f"Testing rezeq integrate - fun {fname} "\
                +f"ntry={ntry} nu={nu:0.2f}")

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
        niter, approx = [0], [s0]
        s_start = s0
        start_exec = time.time()
        for i in range(len(t1)-1):
            t_start = t1[i]
            #print(f"\n### [{i:3d}] t_start = {t_start} / s_start = {s_start:0.5f}###")
            delta = t1[i+1]-t_start
            n, s_end, _ = rezeq.integrate(alphas, scalings, nu, \
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

