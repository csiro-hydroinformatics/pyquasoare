from pathlib import Path
import math
import re
import pytest

import numpy as np
import pandas as pd

from hydrodiy.io import iutils

from hydrodiy.io import csv

from pyquasoare import approx, steady, benchmarks

from test_approx import generate_samples

import data_reader

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent

LOGGER = iutils.get_logger("steady", flog=FTEST / "test_steady.log")

def test_kahan(allclose):
    # Implementing tests by Kayan
    # https://people.eecs.berkeley.edu/%7Ewkahan/Qdrtcs.pdf
    #a1, b1, c1 = 10.27, 29.61, 85.37
    #stdy1 = steady.quad_steady(a1, -2.*b1, c1)

    #a2, b2, c2 = 10.28, 29.62, 85.34
    #stdy2 = steady.quad_steady(a2, -2.*b2, c2)

    #a3, b3, c3 = 94906265.625, 94906267.000, 94906268.375
    #stdy3 = steady.quad_steady(a3, -2.*b3, c3)

    #a4, b4, c4 = 94906266.375, 94906267.375, 94906268.375
    #stdy4 = steady.quad_steady(a4, -2.*b4, c4)

    # Fibonacci numbers
    Fnm1, Fnm2 = 1, 0
    print("\n")
    for n in range(2, 70):
        Fn = Fnm1 + Fnm2

        # Root of Fn.x^2-2Fn-1.x+Fn-2 = 0
        In = 1 if n%2 == 0 else 1j
        x1 = (Fnm1 - In) / Fn
        x2 = (Fnm1 + In) / Fn

        f1 = abs(Fn * x1 * x1 - 2 * Fnm1 * x1 + Fnm2)
        f2 = abs(Fn * x2 * x2 - 2 * Fnm1 * x2 + Fnm2)

        # Test steady
        coefs = np.array([float(Fn), -2.*float(Fnm1), float(Fnm2)])
        stdy = steady.quad_steady(coefs)
        x1s, x2s = stdy[0]

        if n % 2 == 0 and n > 2:
            assert np.isclose(x1s, x1)
            if n < 42:
                assert np.isclose(x2s, x2)
        # loop
        Fnm2 = Fnm1
        Fnm1 = Fn


def test_steady_state(allclose, generate_samples):
    cname, case, params, _, _ = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("Skip param config")

    stdy = steady.quad_steady(params)

    allgood = np.all(~np.isnan(stdy), axis=1)
    if allgood.sum()>0:
        assert np.all(stdy[allgood, 1]>stdy[allgood, 0])

    f1 = np.array([approx.quad_fun(p, s[0]) for p, s in zip(params, stdy)])
    f2 = np.array([approx.quad_fun(p, s[1]) for p, s in zip(params, stdy)])

    a, b, c = params.T
    ina = np.array([approx.isnull(aa)==1 for aa in a])
    nnb = np.array([approx.notnull(aa)==1 for aa in a])
    ilin = ina & nnb
    if ilin.sum()>0:
        assert allclose(f1[ilin], 0)
        assert np.all(np.isnan(stdy[ilin, 1]))

    delta = b**2 - 4*a*c
    nna = np.abs(a)>1e-14 # Caution with very low values of a !
    ipos = (delta>0) & nna
    if ipos.sum()>0:
        assert allclose(f1[ipos], 0, atol=5e-6)
        assert allclose(f2[ipos], 0, atol=5e-6)

    ind = np.array([approx.isnull(d)==1 for d in delta])
    izero = ind & nna
    if izero.sum()>0:
        az, bz, cz = a[izero], b[izero], c[izero]
        rz = stdy[izero, 0]
        assert allclose(f1[izero], 0, atol=5e-6)
        assert np.all(np.isnan(stdy[izero, 1]))


def test_flux_scalings(allclose, generate_samples):
    cname, case, params, _, _ = generate_samples
    ntry = len(params)

    alphas = np.array([-1., 0., 1.])
    flux_scalings = np.random.uniform(0, 1, size=(5, 1))
    tested = 0
    for coefs in params:
        # Repeat the same coefs for the two interpolation bands
        co = np.array([coefs] * 2)[None, :, :]

        # Compute solutions
        stdy = steady.quad_steady_flux_scalings(alphas, co, flux_scalings)
        for i in range(stdy.shape[0]):
            if all(np.isnan(stdy[i])):
                continue

            sc = flux_scalings[i]
            a, b, c = coefs * sc

            for j in range(stdy.shape[1]):
                s = stdy[i, j]
                if np.isnan(s):
                    continue

                if s < alphas[0]:
                    b0 = 2 * a
                    c0 = a * alphas[0]**2 + (b - 2 * a) * alphas[0] + c
                    f = b0 * s + c0
                elif s > alphas[-1]:
                    b0 = 2 * a
                    c0 = a * alphas[-1]**2 + (b - 2 * a) * alphas[-1] + c
                    f = b0 * s + c0
                else:
                    f = a * s**2 + b * s + c

                assert allclose(f, 0, atol=1e-10)

    LOGGER.info(f"[{case}:{cname}] steady flux_scalings: tested={(100.*tested)/ntry:0.0f}%")


def test_flux_scalings_extrapolation(allclose):
    al0, al1, al2 = 0., 1., 2.
    alphas = np.array([al0, al1, al2])
    falphas = np.array([1., 2., 1.])
    fmid = np.array([3., 2.5])
    coefs = approx.quad_coefficients(alphas, falphas, fmid, 2)[None, :, :]
    flux_scalings = np.ones((3, 1))

    stdy = steady.quad_steady_flux_scalings(alphas, coefs, flux_scalings)

    import matplotlib.pyplot as plt
    xx = np.linspace(al0-1, al2 + 1, 1000)
    yy = approx.quad_fun_from_matrix(alphas, coefs, xx)
    plt.plot(xx, yy)
    plt.plot(alphas, falphas, "o")
    plt.plot(stdy[0], [0] * stdy.shape[1], "+")
    plt.grid()
    plt.show()
    import pdb; pdb.set_trace()


    assert allclose(stdy[:, 0], stdy[0, 0])
    #assert allclose(stdy[:, 1], stdy[0, 1])

    # steady state in extrapolationt
    assert stdy[0, 0] > al2

    feval = approx.quad_fun_from_matrix(alphas, coefs, stdy[0])

    import pdb; pdb.set_trace()


    assert allclose(feval, 0.)


def test_flux_scalings_gr4j(allclose):
    nalphas = 25
    alphas = 0.05*np.arange(nalphas)

    fluxes, _ = benchmarks.gr4jprod_fluxes_noscaling()
    coefs = approx.quad_coefficient_matrix(fluxes, alphas)

    nval = 200
    flux_scalings = np.random.uniform(0, 100, size=(nval, 3))
    flux_scalings[:, -1] = 1
    stdy = steady.quad_steady_flux_scalings(alphas, coefs, flux_scalings)

    # only one steady state
    assert stdy.shape[1] == 1

    for t in range(nval):
        s0 = stdy[t, 0]
        # Check steady on approx fun
        co = coefs * flux_scalings[t][None, :]
        out = approx.quad_fun_from_matrix(alphas, co, s0)
        fsum = out.sum(axis=1)
        assert allclose(fsum[~np.isnan(fsum)], 0.)

        # Check steady on original fun
        feval = np.array([f(s0)*flux_scalings[t, ifun] for ifun, f in enumerate(fluxes)])
        fsum = np.sum(feval, axis=0)
        assert allclose(fsum[~np.isnan(fsum)], 0., atol=1e-6)



