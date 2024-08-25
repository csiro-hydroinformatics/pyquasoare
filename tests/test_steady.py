from pathlib import Path
import math
import re
import pytest

import numpy as np
import pandas as pd

from hydrodiy.io import iutils

from hydrodiy.io import csv

from pyrezeq import approx, steady, benchmarks

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
        Fn = Fnm1+Fnm2

        # Root of Fn.x^2-2Fn-1.x+Fn-2 = 0
        In = 1 if n%2 == 0 else 1j
        x1 = (Fnm1-In)/Fn
        x2 = (Fnm1+In)/Fn

        f1 = abs(Fn*x1*x1-2*Fnm1*x1+Fnm2)
        f2 = abs(Fn*x2*x2-2*Fnm1*x2+Fnm2)

        # Test steady
        stdy = steady.quad_steady(float(Fn), -2.*float(Fnm1), float(Fnm2))
        x1s, x2s = stdy

        if n%2 == 0 and n>2:
            assert np.isclose(x1s, x1)
            if n<42:
                assert np.isclose(x2s, x2)
        # loop
        Fnm2 = Fnm1
        Fnm1 = Fn


def test_steady_state(allclose, generate_samples):
    cname, case, params, _, _ = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("Skip param config")

    a, b, c = [np.ascontiguousarray(v) for v in params.T]
    stdy = steady.quad_steady(a, b, c)

    allgood = np.all(~np.isnan(stdy), axis=1)
    if allgood.sum()>0:
        assert np.all(stdy[allgood, 1]>stdy[allgood, 0])

    f1 = approx.quad_fun(a, b, c, stdy[:, 0])
    f2 = approx.quad_fun(a, b, c, stdy[:, 1])

    ina = np.array([approx.isnull(aa)==1 for aa in a])
    nnb = np.array([approx.notnull(aa)==1 for aa in a])
    ilin = ina & nnb
    if ilin.sum()>0:
        assert allclose(f1[ilin], 0)
        assert np.all(np.isnan(stdy[ilin, 1]))

    delta = b**2-4*a*c
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



def test_scalings(allclose, generate_samples):
    cname, case, params, _, _ = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("Skip param config")

    stdy, feval = [], []
    alphas = np.array([-np.inf, 0, np.inf])
    scalings = np.ones((3, 1))
    tested = 0
    for a, b, c in params:
        amat, bmat, cmat = [np.ones((2, 1))*v for v in [a, b, c]]
        stdy = steady.quad_steady_scalings(alphas, scalings, \
                                            amat, bmat, cmat)
        stdy0 = steady.quad_steady(a, b, c)
        notnan = ~np.isnan(stdy0)
        if stdy.shape[1]>0 or notnan.sum()>0:
            tested += 1
            # All values are identical in the 0 axis
            # because the scalings are identical
            assert allclose(np.diff(stdy, axis=0), 0.)
            stdy = stdy[0]

            # Compare with simple steady computation
            assert allclose(stdy, stdy0[~np.isnan(stdy0)])

            # Check steady state value is 0
            feval = approx.quad_fun(a, b, c, stdy)
            if abs(a)>1e-14:
                assert allclose(feval, 0, atol=5e-5)

    LOGGER.info(f"[{case}:{cname}] steady scalings: tested={(100.*tested)/ntry:0.0f}%")



def test_scalings_gr4j(allclose):
    nalphas = 25
    alphas = 0.05*np.arange(nalphas)

    fluxes, _ = benchmarks.gr4jprod_fluxes_noscaling()
    amat, bmat, cmat = approx.quad_coefficient_matrix(fluxes, alphas)

    nval = 200
    scalings = np.random.uniform(0, 100, size=(nval, 3))
    scalings[:, -1] = 1

    stdy = steady.quad_steady_scalings(alphas, scalings, amat, bmat, cmat)

    # only one steady state
    assert stdy.shape[1] == 1

    for t in range(nval):
        s0 = stdy[t, 0]
        # Check steady on approx fun
        amats = amat*scalings[t][None, :]
        bmats = bmat*scalings[t][None, :]
        cmats = cmat*scalings[t][None, :]
        out = approx.quad_fun_from_matrix(alphas, amats, bmats, cmats, s0)
        fsum = out.sum(axis=1)
        assert allclose(fsum[~np.isnan(fsum)], 0.)

        # Check steady on original fun
        feval = np.array([f(s0)*scalings[t, ifun] for ifun, f in enumerate(fluxes)])
        fsum = np.sum(feval, axis=0)
        assert allclose(fsum[~np.isnan(fsum)], 0., atol=1e-6)



def test_steady_shooting(allclose):
    nalphas = 20
    alphas = np.linspace(0., 1.2, nalphas)
    start, end = "2017-01", "2022-12-31"
    nsubdiv = 50000
    X1s = [50, 200, 1000]
    LOGGER.info("")

    # Compute approx coefs
    fluxes, _ = benchmarks.gr4jprod_fluxes_noscaling()
    amat, bmat, cmat = approx.quad_coefficient_matrix(fluxes, alphas)

    # Loop over sites
    for isite, siteid in enumerate(data_reader.SITEIDS):
        # Get climate data
        df = data_reader.get_data(siteid, "daily")
        df = df.loc[start:end]
        nval = len(df)
        inputs = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]]

        X1 = 200
        scalings = (inputs-inputs.iloc[:, [1, 0]].values).clip(0.)/X1
        scalings = np.column_stack([scalings, np.ones(len(scalings))])
        scalings = pd.DataFrame(scalings, index=inputs.index)

        means = scalings.rolling("30D", closed="left").mean()\
                    .groupby(inputs.index.dayofyear).mean().iloc[:-1]\
                    .values

        s0 = 0.5
        nit, s1, fx = steady.quad_steady_scalings_shooting(alphas, means, \
                                amat, bmat, cmat, s0, 1.)

        import matplotlib.pyplot as plt
        plt.plot(s1)
        plt.show()
        import pdb; pdb.set_trace()



