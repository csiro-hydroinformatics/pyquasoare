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


def test_steady_state(allclose, generate_samples):
    cname, case, params, nus, _, _ = generate_samples
    ntry = len(params)
    if ntry==0:
        pytest.skip("Skip param config")

    stdy, feval = [], []
    a, b, c = [np.ascontiguousarray(v) for v in params.T]
    ones = np.ones(len(a))
    for nu in nus:
        s = steady.steady_state(nu, a, b, c)
        f = np.column_stack([approx.approx_fun(nu, a, b, c, sc)\
                                for sc in s.T])
        stdy.append(s)
        feval.append(f)

    stdy = np.row_stack(stdy)
    feval = np.row_stack(feval)

    if case<4:
        # No steady state
        assert np.all(np.isnan(stdy))
        return

    if np.all(np.isnan(stdy)):
        pytest.skip("No steady state found")

    if case>=8:
        # 2 distinct roots
        Delta = a**2-4*b*c
        ipos = np.repeat(Delta>0, ntry)
        iboth = np.sum(~np.isnan(stdy), axis=1)==2
        assert np.all(np.diff(stdy[iboth&ipos], axis=1)>0)

    ione = np.isnan(stdy).sum(axis=1)==1
    nbnan = np.isnan(stdy[ione, 0]).sum()
    errmess = f"Number of nan in first column = {nbnan}"
    assert nbnan == 0, LOGGER.error(errmess)

    # check steady state
    err_max = np.nanmax(np.abs(feval))
    assert err_max < 5e-4

    nskipped = np.all(np.isnan(feval), axis=1).sum()
    mess = f"[{case}:{cname}] steady: errmax = {err_max:3.2e}"\
            f"  skipped={(100.*nskipped)/len(feval):0.0f}%"
    LOGGER.info(mess)


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
    nu, amat, bmat, cmat, niter, fopt = approx.optimize_nu(funs, alphas)

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


