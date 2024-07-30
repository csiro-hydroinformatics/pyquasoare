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

    LOGGER.info("")
    err_max = 0
    a, b, c = [np.ascontiguousarray(v) for v in params.T]
    stdy = steady.steady_state(nus, a, b, c)

    if case<4:
        # No steady state
        assert np.all(np.isnan(stdy))
        pytest.skip("No steady state for this case")

    if np.all(np.isnan(stdy)):
        pytest.skip("No steady state found")

    # check nan values
    iboth = np.isnan(stdy).sum(axis=1)==0
    assert np.all(np.diff(stdy[iboth], axis=1)>=0)

    if case>=8:
        # 2 distinct roots
        Delta = a**2-4*b*c
        ipos = Delta>0
        assert np.all(np.diff(stdy[iboth&ipos], axis=1)>0)

    ione = np.isnan(stdy).sum(axis=1)==1
    assert np.all(~np.isnan(stdy[ione, 0]))

    # check steady state
    f = np.array([[approx.approx_fun(nu, aa, bb, cc, s) for nu, aa, bb, cc, s\
                        in zip(nus, a, b, c, ss)] for ss in stdy.T]).T
    err_max = np.nanmax(np.abs(f))
    assert err_max < 5e-4

    nskipped = np.all(np.isnan(f), axis=1).sum()
    mess = f"steady - Case {cname}: errmax = {err_max:3.2e}"\
            f"  skipped={100*(nskipped)/ntry:0.0f}%"
    LOGGER.info(mess)


def test_steady_state_scalings(allclose):
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

    stdy = steady.steady_state_scalings(alphas, nu, scalings, amat, bmat, cmat)
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


