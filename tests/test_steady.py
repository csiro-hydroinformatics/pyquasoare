from pathlib import Path
import math
import re
import pytest

import numpy as np
import pandas as pd

from hydrodiy.io import iutils

from hydrodiy.io import csv

from pyrezeq import approx, steady

from test_approx import generate_samples

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent

LOGGER = iutils.get_logger("steady", flog=FTEST / "test_steady.log")

def test_steady_state_kahan(allclose):
    # Implementing tests by Kayan
    # https://people.eecs.berkeley.edu/%7Ewkahan/Qdrtcs.pdf
    a1, b1, c1 = 10.27, 29.61, 85.37
    stdy1 = steady.quad_steady(a1, -2.*b1, c1)

    D1 = b1*b1-a1*c1
    za1 = (b1+np.sign(b1)*math.sqrt(D1))/a1
    zb1 = (b1-np.sign(b1)*math.sqrt(D1))/a1

    a2, b2, c2 = 10.28, 29.62, 85.34
    stdy2 = steady.quad_steady(a2, -2.*b2, c2)

    a3, b3, c3 = 94906265.625, 94906267.000, 94906268.375
    stdy3 = steady.quad_steady(a3, -2.*b3, c3)

    a4, b4, c4 = 94906266.375, 94906267.375, 94906268.375
    stdy4 = steady.quad_steady(a4, -2.*b4, c4)
    import pdb; pdb.set_trace()



def test_steady_state(allclose, generate_samples):
    cname, case, params, _, _ = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("Skip param config")

    a, b, c = [np.ascontiguousarray(v) for v in params.T]
    stdy = steady.quad_steady(a, b, c)
    f1 = approx.quad_fun(a, b, c, stdy[:, 0])
    f2 = approx.quad_fun(a, b, c, stdy[:, 1])

    ina = np.array([approx.isnull(aa)==1 for aa in a])
    nnb = np.array([approx.notnull(aa)==1 for aa in a])
    ilin = ina & nnb
    if ilin.sum()>0:
        assert allclose(f1[ilin], 0)
        assert np.all(np.isnan(stdy[ilin, 1]))

    delta = b**2-4*a*c
    nna = ~ina
    ipos = (delta>0) & nna
    if ipos.sum()>0:
        assert allclose(f1[ipos], 0)
        assert allclose(f2[ipos], 0)

    ind = np.array([approx.isnull(d)==1 for d in delta])
    izero = ind & nna
    if izero.sum()>0:
        az, bz, cz = a[izero], b[izero], c[izero]
        rz = stdy[izero, 0]
        assert allclose(f1[izero], 0)
        assert np.all(np.isnan(stdy[izero, 1]))



def test_steady_state_scalings(allclose, generate_samples):
    cname, case, params, nus, _, _ = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("Skip param config")

    stdy, feval = [], []
    alphas = np.array([-np.inf, 0, np.inf])
    scalings = np.ones((3, 1))
    tested = 0
    for nu, (a, b, c) in zip(nus, params):
        amat, bmat, cmat = [np.ones((2, 1))*v for v in [a, b, c]]
        stdy, bands = steady.steady_state_scalings(alphas, nu, scalings, \
                                            amat, bmat, cmat)
        if case<4:
            # No steady state
            assert stdy.shape[1]==0
        else:
            stdy0 = steady.steady_state(nu, a, b, c)
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
                feval = approx.approx_fun(nu, a, b, c, stdy)
                assert allclose(feval, 0, atol=5e-5)

    LOGGER.info(f"[{case}:{cname}] steady scalings: tested={(100.*tested)/ntry:0.0f}%")


def test_steady_state_scalings_gr4j(allclose):
    nalphas = 7
    alphas = np.linspace(0, 1.2, nalphas)
    # GR4J production
    funs = [
        lambda x: 1-x**2, \
        lambda x: -x*(2-x), \
        lambda x: -(4/9*x)**5/4
    ]
    scr = np.ones(3)
    nu, amat, bmat, cmat, niter, fopt = approx.optimize_nu(funs, alphas, scr)

    nval = 200
    scalings = np.random.uniform(0, 100, size=(nval, 3))
    scalings[:, -1] = 1

    stdy, bands = steady.steady_state_scalings(alphas, nu, scalings, amat, bmat, cmat)

    for t in range(nval):
        s0 = stdy[t]
        # Check steady on approx fun
        amats = amat*scalings[t][None, :]
        bmats = bmat*scalings[t][None, :]
        cmats = cmat*scalings[t][None, :]
        out = approx.approx_fun_from_matrix(alphas, nu, amats, bmats, cmats, s0)
        fsum = out.sum(axis=1)
        assert allclose(fsum[~np.isnan(fsum)], 0.)

        # Check steady on original fun
        feval = np.array([f(s0)*scalings[t, ifun] for ifun, f in enumerate(funs)])
        fsum = np.sum(feval, axis=0)
        assert allclose(fsum[~np.isnan(fsum)], 0., atol=1e-3)


