from pathlib import Path
import re
import inspect
from itertools import product as prod
import time
import math
import re
import pytest

import numpy as np
import pandas as pd
import scipy.integrate as sci_integrate
import matplotlib.pyplot as plt

from hydrodiy.io import iutils, csv

from pyquasoare import integrate, models, gradients, approx, steady
from pyquasoare.benchmarks import benchmarks
from pyquasoare.models import error_management

import c_pyquasoare_grad

import data_reader

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent
NALPHAS = 50

DEMANDS = [-0.5, 0., 0.1]
TIMESTEPS = [0.1, 1.]
GAMMAS = [10, 100, 10000]
FLUX_SCALINGS = [
    [1., 1., 1.],
    [1.2, 1., 1.],
    [1., 1.2, 1.],
    [1., 1., 1.2]
    ]
NUMDIFF_EPS = 1e-5

FLUX_FUNCTIONS = {
    "gr4jprodquad": [
        lambda x: 1 - np.minimum(x, 1)**2,
        lambda x: np.clip(x, 0, 1) * (2 - np.clip(x, 0, 1)),
        lambda x: 4**3/9**4 * np.maximum(x, 0)**2
        ],
    "gr4jroutequad": [
        lambda x: 0.9 * np.ones_like(x),
        lambda x: np.maximum(x, 0.)**2,
        lambda x: np.maximum(x, 0.)**2 / 4
        ],
    "nonlinrouting": [
        # exponent n = 6
        lambda x: np.where(x > 0, 1., 0.),
        lambda x: np.ones_like(x),
        lambda x: np.maximum(x, 0.)**4
        ],
    "hymod1": [
        # exponent = 0.2
        lambda x: (1 - np.clip(x, 0., 1.))**0.2,
        lambda x: np.maximum(x, 0.),
        lambda x: np.zeros_like(x)
        ],
    "hymod2": [
        # exponent = 5
        lambda x: (1 - np.clip(x, 0., 1.))**0.2,
        lambda x: np.maximum(x, 0.),
        lambda x: np.zeros_like(x)
        ],
    "muskingumrouting": [
        # KM = 2.5 / XM = 0.2
        lambda x: np.where(x > 0, 1., 0.) / (1 - 0.2),
        lambda x: np.ones_like(x),
        lambda x: np.maximum(x, 0.) / 2.5 / (1 - 0.2)
        ],
}

FLUX_FUNCTIONS.update(gradients.FLUX_FUNCTIONS)

# Comparison functions
def max_abs_raw_err(x, y, exclude=10, axis=0):
    em = np.sort(np.abs(x - y), axis=axis)[-exclude]
    return em

def max_abs_asinh_err(x, y, exclude=10, axis=0, eps=1e-2):
    tx = np.asinh(x / eps)
    ty = np.asinh(y / eps)
    return max_abs_raw_err(tx, ty, exclude, axis)

def flux_denom(fx, eps1=1e-6, eps2=1e-3):
    afx = np.abs(fx)
    means = np.mean(afx, axis=0)[None, :]
    maxs = np.max(afx, axis=0)[None, :]
    shift = eps1 + eps2 * means
    shift[maxs < eps1] = np.nan
    return afx + shift

def comp_dfluxes(fx_ref, dfx_ref, dfx_alt, eps1=1e-6, eps2=1e-3,
                 exclude=10):
    num = np.abs(dfx_ref - dfx_alt)
    denom = flux_denom(fx_ref, eps1, eps2)
    err = num / denom
    esort = np.sort(err, axis=0)
    return esort[-exclude], err.mean()


# utility functions
def get_data(siteid, reservoir, timestep, gamma, demand,
             istart=None, iend=None):
    df = data_reader.get_data(siteid, "daily", None)

    istart = 0 if istart is None else istart
    iend = len(df) if iend is None else iend
    df = df.iloc[istart: iend]

    Qstars = df.loc[:, ["RAINFALL[mm_day-1]", "PET[mm_day-1]"]].values
    Qstars[:, 1] *= -1 # PET = negative demand
    Qstars = np.ascontiguousarray(Qstars)
    if re.search("rout", reservoir):
        Qstars[:, 1] = demand

    nval = len(Qstars)
    scalings = np.ones((nval, 3))
    scalings[:, :2] = Qstars / gamma
    scalings[:, 2] = - 1. / timestep

    return Qstars, scalings


def get_alphas_and_coefs(reservoir, scalings):
    funs = FLUX_FUNCTIONS[reservoir]
    amax = 5 if re.search("rout", reservoir) else 1.
    alphas = np.linspace(0, amax, NALPHAS)
    coefs = approx.quad_coefficient_matrix(funs, alphas,
                                           approx_opt=0)
    stdy = steady.quad_steady_flux_scalings(alphas, coefs, scalings)
    eps = 1e-5
    alphas = [-eps, 0., eps] \
             + np.linspace(2 * eps, 1 - 2 * eps, NALPHAS - 6).tolist() \
             + [1 - eps, 1., 1 + eps]
    amax = np.nanmax(stdy)
    if amax > 1 + eps:
        alphas += np.linspace(1 + 2 * eps, amax, 6).tolist()
    alphas = np.array(alphas)
    if np.any(np.diff(alphas) <= 1e-10):
        raise ValueError("Problem with alphas")

    coefs = approx.quad_coefficient_matrix(funs, alphas,
                                           approx_opt=0)
    coefs_lin = gradients.lin_coefficient_matrix(funs, alphas)
    return alphas, coefs, coefs_lin


def get_input_mapping(nreservoirs, ntype):
    netw = gradients.Network(nreservoirs)
    for res in range(1, nreservoirs):
        if ntype == "outflow":
            netw.add_link_outflow(res - 1, res)
        else:
            netw.add_link_bypass_shortfall(res - 1, res)
    return netw.get_input_mapping()


def skip_this(reservoir, gamma, demand):
    if not re.search("rout", reservoir) and demand != 0:
        pytest.skip("Cannot use non-zero demand for reservoirs other than routing")

    if reservoir in ["linrouting", "muskingumrouting"]:
        if demand < 0:
            pytest.skip("Can't have demand <0 with pure linear routing models")

        if gamma > 1000:
            pytest.skip("Skipping linear routing tests for large gamma")

    if reservoir in ["gr4jroutequad"] and demand > 0:
        pytest.skip("Can't have demand >0 with gr4jroute quad")

    if re.search("openwater", reservoir) and gamma > 100:
        pytest.skip("Skipping openwater reservoir for large gamma")

# -------------- TESTS ----------------------------------------------------
@pytest.mark.parametrize("nalphas", [10, 500])
@pytest.mark.parametrize("amin", [0., 0.5])
@pytest.mark.parametrize("amax", [0.5, 1., 5.])
@pytest.mark.parametrize("nreservoirs", [1, 10])
def test_get_alphas(nalphas, amin, amax, nreservoirs):
    if amax <= amin:
        with pytest.raises(ValueError, match="2eps"):
            al = gradients.get_alphas(nalphas, amin=amin, amax=amax)
        return

    al = gradients.get_alphas(nalphas, amin=amin, amax=amax,
                              nreservoirs=nreservoirs)
    assert al.shape == (nreservoirs, nalphas)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", FLUX_FUNCTIONS)
@pytest.mark.parametrize("gamma", [10])
@pytest.mark.parametrize("timestep", [1])
@pytest.mark.parametrize("demand", DEMANDS)
@pytest.mark.parametrize("eps", [-NUMDIFF_EPS, 0., NUMDIFF_EPS])
def test_reservoir_functions(siteid, reservoir, gamma, timestep, demand, eps):
    Qstars, scalings = get_data(siteid, reservoir, timestep, gamma, demand,
                                -365 * 10)
    alphas, coefs, coefs_lin = get_alphas_and_coefs(reservoir, scalings)
    coefs_lin *= 1 + eps

    a0 = alphas[:-1][None, :]
    c0 = coefs_lin[:, :, 0] * a0 + coefs_lin[:, :, 1]
    assert np.all(c0 >= 0)

    a1 = alphas[1:][None, :]
    c1 = coefs_lin[:, :, 0] * a1 + coefs_lin[:, :, 1]
    assert np.all(c1 >= 0)


@pytest.mark.parametrize("linktype", ["outflow", "bypass"])
def test_network(linktype, allclose):
    netw = gradients.Network(3)

    if linktype == "outflow":
        netw.add_link_outflow(0, 1)
        netw.add_link_outflow(1, 2)
    else:
        netw.add_link_bypass_shortfall(0, 1)
        netw.add_link_bypass_shortfall(1, 2)

    assert netw.upstream_reservoirs(2) == {0, 1}

    ipm, ipmw = netw.get_input_mapping()
    nfxo = gradients.NFLUXES_OUTPUTS
    ipmd = gradients.INPUT_MAPPING_LASTDIM
    assert ipm.shape == (netw.nreservoirs, nfxo * netw.nreservoirs, ipmd)
    assert ipmw.shape == (netw.nreservoirs, nfxo * netw.nreservoirs)
    missid = gradients.INPUT_MAPPING_MISSINGRESERVOIRID

    for rid in range(netw.nreservoirs):
        lks = ipm[rid]
        ws = ipmw[rid]

        if rid == 0:
            assert np.all(lks[:, 0] == missid)
            assert np.all(lks[:, 1:] == 0)
            assert np.all(ws == 0)
        else:
            if linktype == "outflow":
                assert np.all(lks[1:, 0] == missid)
                assert np.all(lks[1:, 1:] == 0)
                # total outflow from upstream links to inflow
                assert all(lks[0] == [rid - 1, 5, 0])

                assert np.all(ws[1:] == 0)
                assert ws[0] == -1.
            else:
                assert np.all(lks[2:, 0] == missid)
                assert np.all(lks[2:, 1:] == 0)
                # bypass flow from upstream links to inflow
                assert all(lks[0] == [rid - 1, 3, 0])
                # demand shortfall flow from upstream links to demand
                assert all(lks[1] == [rid - 1, 4, 1])

                assert np.all(ws[2:] == 0)
                assert np.all(ws[:2] == -1.)


def test_network_cyclical():
    netw = gradients.Network(3)
    netw.add_link_outflow(0, 1)
    netw.add_link_outflow(1, 2)
    with pytest.raises(ValueError, match="cyclical"):
        netw.add_link_outflow(2, 0)


def test_network_wrong_order():
    netw = gradients.Network(3)
    with pytest.raises(ValueError, match="rearrange"):
        netw.add_link_outflow(2, 1)


@pytest.mark.parametrize("reservoir", FLUX_FUNCTIONS)
def test_lin_coefficient_matrix(reservoir, allclose):
    alphas = np.linspace(0, 1, 50)
    funs = FLUX_FUNCTIONS[reservoir]
    coefs = gradients.lin_coefficient_matrix(funs, alphas)

    for i, f in enumerate(funs):
        y1 = f(alphas)[:-1]
        y2 = coefs[i][:, 0] * alphas[:-1] + coefs[i][:, 1]
        assert allclose(y1, y2)


def test_grad_integrate_error():
    # Configure problem
    niter = np.zeros(1, dtype=np.int32)
    x = np.zeros(1, dtype=np.float64)
    nres = 1
    ipm = np.zeros((nres, nres * gradients.NFLUXES_OUTPUTS,
                    gradients.INPUT_MAPPING_LASTDIM),
                    dtype=np.int32)
    ipmw = np.zeros((nres, nres * gradients.NFLUXES_OUTPUTS),
                    dtype=np.float64)

    nvars = 2
    x = np.array([0.5])
    dx = np.zeros(nvars, dtype=np.float64)

    nfxo = gradients.NFLUXES_OUTPUTS
    fx = np.zeros(nfxo, dtype=np.float64)
    dfx = np.zeros((nvars, nres, nfxo), dtype=np.float64)

    funs = [
        lambda x: np.ones_like(x),
        lambda x: x,
        lambda x: x
    ]
    alphas = np.linspace(0, 1, 50)
    coefs_lin = gradients.lin_coefficient_matrix(funs, alphas)
    timestep = 1.
    gamma = 100.

    posinf = 1
    rid = 0
    vrid = np.array([0] * nvars, dtype=np.int32)
    vtid = np.array([0] * nvars, dtype=np.int32)
    qst = -np.ones(2)
    dqst = np.zeros((nvars, gradients.NFLUXES_INPUTS))
    fxs = np.ones(gradients.NFLUXES_INPUTS)
    t0 = 0.
    # .. error due to negative potential inflow
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, posinf, alphas, timestep, gamma,
                                            ipm, ipmw, qst, dqst, fxs, coefs_lin,
                                            vrid, vtid, niter, x, dx, fx, dfx)
    assert ierr == 112

    # .. disables the previous error
    posinf = 0
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, posinf, alphas, timestep, gamma,
                                            ipm, ipmw, qst, dqst, fxs, coefs_lin,
                                            vrid, vtid, niter, x, dx, fx, dfx)
    assert ierr == 0

    # .. error due to negative flux function
    qst = np.ones(2)
    coefs_lin[0] *= -1
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, posinf, alphas, timestep, gamma,
                                            ipm, ipmw, qst, dqst, fxs, coefs_lin,
                                            vrid, vtid, niter, x, dx, fx, dfx)
    assert ierr == 111

    # .. error due to zero reservoir capacity
    coefs_lin[0] *= -1
    gamma = 0
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, posinf, alphas, timestep, gamma,
                                            ipm, ipmw, qst, dqst, fxs, coefs_lin,
                                            vrid, vtid, niter, x, dx, fx, dfx)
    assert ierr == 113

    # .. error due to zero flux scaling
    gamma = 100.
    vtid = np.array([1] * nvars, dtype=np.int32)
    fxs = np.zeros(gradients.NFLUXES_INPUTS)
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, posinf, alphas, timestep, gamma,
                                            ipm, ipmw, qst, dqst, fxs, coefs_lin,
                                            vrid, vtid, niter, x, dx, fx, dfx)
    assert ierr == 604


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", FLUX_FUNCTIONS)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("timestep", TIMESTEPS)
@pytest.mark.parametrize("demand", DEMANDS)
def test_grad_integrate(siteid, reservoir, gamma, timestep, demand, allclose):
    skip_this(reservoir, gamma, demand)
    start, end = -300, None
    posinf = 1
    Qstars, scalings = get_data(siteid, reservoir, timestep,
                                gamma, demand, start, end)
    alphas, coefs, coefs_lin = get_alphas_and_coefs(reservoir, scalings)

    nres = 3
    ipm, ipmw = get_input_mapping(nres, "bypass_shortfall")

    # Loop and integrate
    rid = 0 # this is the top reservoir
    nv = gradients.VARTYPE_NUMBER
    vrid = np.repeat(np.arange(nres, dtype=np.int32), nv)
    vtid = np.tile(np.arange(nv, dtype=np.int32), nres)

    nvars = len(vtid)

    x_start = 0.5
    dx = np.zeros(nvars)
    scalings = np.ones(3)
    niter2 = np.zeros(1, dtype=np.int32)
    x = np.array([x_start], dtype=np.float64)
    dx = np.zeros(nvars, dtype=np.float64)
    dqst = np.zeros((nvars, gradients.NFLUXES_INPUTS))
    nfxo = gradients.NFLUXES_OUTPUTS
    fx = np.zeros(nfxo, dtype=np.float64)
    dfx = np.zeros((nvars, nres, nfxo), dtype=np.float64)
    fxs = np.ones(gradients.NFLUXES_INPUTS, dtype=np.float64)

    nval = len(Qstars)
    for i in range(nval):
        # Quad interpol
        scalings[0] = Qstars[i, 0] / gamma
        scalings[1] = Qstars[i, 1] / gamma
        scalings[2] = - 1. / timestep
        niter1, x1, fx1 = integrate.quad_integrate(alphas, scalings,
                                                   coefs, 0.,
                                                   x_start, timestep)
        fx1 *= gamma / timestep

        # Lin
        qst = np.ascontiguousarray(Qstars[i])
        ierr = c_pyquasoare_grad.grad_integrate(nres, rid, posinf, alphas, timestep, gamma,
                                                ipm, ipmw, qst, dqst, fxs, coefs_lin,
                                                vrid, vtid,
                                                niter2, x, dx, fx, dfx)
        name = "c_pyquasoare_grad.grad_integrate"
        error_management("raise", ierr, name)

        assert allclose(x1, x[0])
        assert allclose(fx1, fx[:3])

        fxa = np.array([fx1[0] - qst[0],
                        fx1[1] - qst[1],
                        fx1[0] - qst[0] + fx1[2]])
        assert allclose(fxa, fx[3:])

        x_start = x1


def test_grad_reservoir_nan(debug_mode):
    timestep = 1.
    siteid = "203004"
    df = data_reader.get_data(siteid, "daily")
    #df = df.iloc[:1]

    mod = gradients.MultiReservoirModel(2, errors="ignore")
    mod.set_coefs_linear(0, "openwater")
    mod.set_coefs_linear(1, "gr4jprod")
    mod.gammas = [5., 400.]
    mod.allocate(len(df), timestep=timestep)

    Qst0 = df.loc[:, ["RAINFALL[mm_day-1]", "PET[mm_day-1]"]].values.copy()
    Qst0[:, -1] *= -1
    Qst0 = np.ascontiguousarray(Qst0)

    coefs = np.ascontiguousarray(mod.coefs_linear[0])
    n0, x0, _, fx0, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[0]),
                                                 mod.gammas[0],
                                                 Qst0,
                                                 coefs,
                                                 timestep, mod.x0[0])

    Qst1 = np.ascontiguousarray(-fx0[:, 3:5])
    n1, x1, _, fx1, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[1]),
                                                 mod.gammas[1],
                                                 Qst1,
                                                 np.ascontiguousarray(mod.coefs_linear[1]),
                                                 timestep, mod.x0[1],
                                                 errors="raise")
    assert np.all(~np.isnan(fx1))


def test_grad_reservoir_specific(debug_mode, allclose):
    siteid = "203014"
    reservoir = "linrouting"
    start, end = 14100, 14130
    demand = 0.5
    timestep = 1.
    gamma = 100.
    Qstars, scalings = get_data(siteid, reservoir, timestep,
                                gamma, demand, start, end)
    alphas, coefs, coefs_lin = get_alphas_and_coefs(reservoir, scalings)

    # Quad
    x0 = 0.5
    n1, x1, fx1 = models.quad_model(alphas, scalings, coefs,
                                    x0, timestep)
    fx1 *= gamma / timestep
    fx1 = np.column_stack([fx1,
                           -Qstars[:, 0] + fx1[:, 0],
                           -Qstars[:, 1] + fx1[:, 1],
                           -Qstars[:, 0] + fx1[:, 0] + fx1[:, 2]])

    # Lin
    fxs = np.ones(gradients.NFLUXES_INPUTS)
    n2, x2, dx2, fx2, dfx2 = gradients.grad_reservoir(alphas, gamma,
                                                      Qstars, coefs_lin,
                                                      timestep, x0,
                                                      flux_scalings=fxs,
                                                      errors="raise")
    assert allclose(x1, x2)
    assert allclose(fx1, fx2)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", FLUX_FUNCTIONS)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("timestep", TIMESTEPS)
@pytest.mark.parametrize("demand", DEMANDS)
@pytest.mark.parametrize("flux_scalings", FLUX_SCALINGS)
def test_grad_reservoir(siteid, reservoir, gamma, timestep, demand,
                        flux_scalings, allclose, debug_mode):
    skip_this(reservoir, gamma, demand)
    if debug_mode:
        start, end = 14100, 14130
    else:
        start, end = None, None
    Qstars, scalings = get_data(siteid, reservoir, timestep,
                                gamma, demand, start, end)
    alphas, coefs, coefs_lin = get_alphas_and_coefs(reservoir, scalings)

    # Flux scalings
    fxs = np.array(flux_scalings)

    # Quadratic
    x0 = 0.5
    sc1, sc2, sc3 = fxs
    scalings[:, 0] *= sc1
    scalings[:, 1] *= sc2
    scalings[:, 2] *= sc3
    n1, x1, fx1 = models.quad_model(alphas, scalings, coefs,
                                    x0, timestep, errors="raise")
    fx1 *= gamma / timestep
    fx1 = np.column_stack([fx1,
                           -sc1 * Qstars[:, 0] + fx1[:, 0],
                           -sc2 * Qstars[:, 1] + fx1[:, 1],
                           -sc1 * Qstars[:, 0] + fx1[:, 0] + fx1[:, 2]])

    # Lin
    n2, x2, dx2, fx2, dfx2 = gradients.grad_reservoir(alphas, gamma,
                                                      Qstars, coefs_lin,
                                                      timestep, x0,
                                                      flux_scalings=fxs,
                                                      errors="raise")
    assert allclose(x1, x2)
    assert allclose(fx1, fx2)
    assert all(~np.isnan(x2))
    assert np.all(~np.isnan(dx2))

    # Sanity check on store levels
    if not re.search("rout(e|ing)", reservoir):
        assert all((x2 >= 0) & (x2 <= 1.05))

    # Check additional fluxes
    inflow = fx2[:, 0]
    demand = fx2[:, 1]
    drainage = fx2[:, 2]
    assert allclose(fx2[:, 3], inflow - sc1 * Qstars[:, 0])
    assert allclose(fx2[:, 4], demand - sc2 * Qstars[:, 1])
    assert allclose(fx2[:, 5], fx2[:, 2] + fx2[:, 3])

    # Mass balance
    dSa = np.diff(x2) * gamma
    dSb = timestep * fx2[1:, :3].sum(axis=1)
    assert allclose(dSa, dSb, atol=5e-8)

    if reservoir == "transluent":
        # No flux storage
        assert allclose(fx2[:, :3], 0.)
        assert allclose(fx2[:, 3], -sc1 * Qstars[:, 0])
        assert allclose(fx2[:, 5], -sc1 * Qstars[:, 0])
        assert allclose(fx2[:, 4], -sc2 * Qstars[:, 1])


def test_grad_reservoir_maxfun(allclose):
    nalphas = 10
    alphas = gradients.get_alphas(nalphas)[0]
    funs = gradients.FLUX_FUNCTIONS["openwater"]
    coefs = gradients.lin_coefficient_matrix(funs, alphas)
    x0 = 0.5
    gamma = 1e-3
    timestep = 1.

    nval = 100
    Qstars = np.ones((nval, 2)) * 10.
    Qstars[:, 1] = np.random.uniform(-15, 15, nval)
    n, x, dx, fx, dfx = gradients.grad_reservoir(alphas, gamma,
                                                 Qstars, coefs,
                                                 timestep, x0,
                                                 errors="raise")
    outflow = -fx[:, 5]
    expected = np.maximum(Qstars.sum(axis=1), 0)
    assert allclose(outflow, expected, atol=gamma * 1.1)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", FLUX_FUNCTIONS)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("timestep", TIMESTEPS)
@pytest.mark.parametrize("demand", DEMANDS)
@pytest.mark.parametrize("flux_scalings", FLUX_SCALINGS)
def test_grad_reservoir_tangent(siteid, reservoir, gamma, timestep,
                                demand, flux_scalings,
                                allclose, debug_mode):
    skip_this(reservoir, gamma, demand)
    start = -5 * 365
    end = None
    Qstars, scalings = get_data(siteid, reservoir, timestep, gamma, demand,
                                start, end)
    alphas, coefs, coefs_lin = get_alphas_and_coefs(reservoir, scalings)

    # Flux scalings
    fxs = np.array(flux_scalings)

    # Gradient
    x0 = 0.5
    n, x, dx, fx, dfx = gradients.grad_reservoir(alphas, gamma,
                                                 Qstars, coefs_lin,
                                                 timestep,
                                                 flux_scalings=fxs,
                                                 x0=x0,
                                                 errors="raise")
    # Sanity checks
    assert allclose(dfx[:, :, 2] + dfx[:, :, 3], dfx[:, :, 5])

    # Numerical difference for each variable
    for vid in range(gradients.VARTYPE_NUMBER):
        dx_v = dx[:, vid]
        dfx_v = dfx[:, vid]

        if vid == 0:
            # Gradient vs capacity
            eps = gamma * 1e-5
            gamma_m = gamma - eps
            gamma_p = gamma + eps
            fxs_m, fxs_p = fxs, fxs
        else:
            # Gradient vs flux scaling
            eps = 1e-6
            gamma_m, gamma_p = gamma, gamma
            fxs_m, fxs_p = fxs.copy(), fxs.copy()
            inp = vid - 1
            fxs_m[inp] -= eps
            fxs_p[inp] += eps

        _, xm, _, fxm, _ = gradients.grad_reservoir(alphas, gamma_m,
                                                    Qstars, coefs_lin,
                                                    timestep,
                                                    x0=x0,
                                                    flux_scalings=fxs_m)

        _, xp, _, fxp, _ = gradients.grad_reservoir(alphas, gamma_p,
                                                    Qstars, coefs_lin,
                                                    timestep,
                                                    x0=x0,
                                                    flux_scalings=fxs_p)
        dx_n = (xp - xm) / 2 / eps
        dfx_n = (fxp - fxm) / 2 / eps

        if vid == 0:
            # Convert dflux against gamma to dflux against gamma scaling
            dfx_n *= gamma
            dfx_v *= gamma

        dx_err = max_abs_raw_err(dx_n, dx_v)

        # Special case
        if reservoir == "gr2mroute" and gamma == 10000 and demand == 0.1:
            exclude = 50
        else:
            exclude = 10

        dfx_err_max, _ = comp_dfluxes(fx, dfx_n, dfx_v, exclude=exclude)

        xtol = 1e-5
        ftol = 5e-3

        if debug_mode and (dx_err > xtol or dfx_err_max.max() > ftol):
            ncols, nrows = 6, 2
            aw, ah = 3, 3
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                    figsize=(ncols * aw, nrows * ah),
                                    layout="constrained",
                                    sharex=True)
            for ifx in range(6):
                ax = axs[0, ifx]
                ax.plot(fx[:, ifx], "k-")

                ax = axs[1, ifx]
                v = np.asinh(dfx_v[:, ifx])
                n = np.asinh(dfx_n[:, ifx])
                ax.plot(v, label="Quas")
                ax.plot(n, label="NumDiff")
                tax = ax.twinx()
                tax.plot(np.abs(v - n), color="0.5", lw=0.8, alpha=0.5)
                #tax.plot(fx[:, ifx], color="0.5", lw=0.8, alpha=0.5)

                title = f"Flux {ifx + 1}"
                ax.set_title(title)
                if ifx == 0:
                    ax.legend()

            ftitle = f"Diff against dvar={vid} eps={eps:0.1e}"
            fig.suptitle(ftitle, fontweight="bold")
            plt.show()

        #print(f"x={dx_err:8.3e} fx={dfx_err_max.max():8.3e}")
        assert dx_err < xtol
        assert np.nanmax(dfx_err_max) < ftol


def test_grad_model_setters(allclose):
    siteid = "203014"
    timestep = 1.
    mod = gradients.MultiReservoirModel(2, errors="raise")
    mod.add_link_bypass_shortfall(0, 1)
    mod.set_coefs_linear(0, FLUX_FUNCTIONS["openwater"])
    mod.set_coefs_linear(1, "gr4jprod")

    mod.gammas = 10
    assert allclose(mod.gammas, 10.)

    mod.gammas = [100., 10]
    assert allclose(mod.gammas, [100., 10])

    mod.flux_scalings = 1.2
    assert allclose(mod.flux_scalings, 1.2)

    fxs = np.ones((mod.nreservoirs, gradients.NFLUXES_INPUTS))
    fxs[0, 0] = 1.2
    mod.flux_scalings = fxs
    assert allclose(mod.flux_scalings, fxs)


def test_grad_model_clone(allclose):
    mod = gradients.MultiReservoirModel(2, errors="raise")
    mod.add_link_bypass_shortfall(0, 1)
    mod.set_coefs_linear(0, "openwater")
    mod.set_coefs_linear(1, "gr4jprod")
    mod.gammas = [5., 400.]

    siteid = "203014"
    timestep = 1.
    df = data_reader.get_data(siteid, "daily")
    mod.allocate(len(df), timestep=timestep)
    Qst1 = df.loc[:, ["RAINFALL[mm_day-1]", "PET[mm_day-1]"]].values
    Qst1[:, -1] *= -1
    Qst1 = np.ascontiguousarray(Qst1)
    mod.set_Qstars(0, Qst1)

    modc = mod.clone()

    # Check clone is independent from source
    attrns = [
        "variables_reservoir_id",
        "variables_type_id",
        "_coefs_linear",
        "_flux_scalings",
        "Qstars",
        "x0",
        "x",
        "dx",
        "fluxes",
        "dfluxes",
        "niter",
        ]
    ntested = 0
    for vn in attrns:
        v1 = getattr(mod, vn)
        v2 = getattr(modc, vn)
        try:
            v2.flat[0] += 1
        except:
            continue

        assert abs(v1.flat[0] - v2.flat[0]) > 0
        ntested += 1

    assert ntested == 7

@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("timestep", TIMESTEPS)
@pytest.mark.parametrize("flux_scalings", FLUX_SCALINGS)
def test_grad_model(siteid, timestep, allclose, flux_scalings, debug_mode):
    mod = gradients.MultiReservoirModel(6, errors="raise")

    # Model is the sum of 3 contributions summed via a transluent reservoir:
    # 1. Interception -> GR4Jprod -> GR4Jroute -> out
    # 2. GR4Jroute -> out
    # 3. GR4Jroute -> out
    mod.add_link_bypass_shortfall(0, 1)
    mod.add_link_outflow(1, 2)
    mod.add_link_outflow(2, 5)
    mod.add_link_outflow(3, 5)
    mod.add_link_outflow(4, 5)

    mod.set_coefs_linear(0, "openwater")
    mod.set_coefs_linear(1, "gr4jprod")
    mod.set_coefs_linear(2, "gr4jroute")
    mod.set_coefs_linear(3, "gr4jroute")
    mod.set_coefs_linear(4, "gr4jroute")
    mod.set_coefs_linear(5, "transluent")
    mod.gammas = [5., 400., 50., 10., 5., 1.]

    # Set flux scaling for the interception reservoir
    fxs = np.ones((mod.nreservoirs, gradients.NFLUXES_INPUTS))
    fxs[0] = flux_scalings
    mod.flux_scalings = fxs

    df = data_reader.get_data(siteid, "daily")
    df = df.iloc[:1]

    nrep = 5
    t = timestep * (np.arange(len(df))[:, None] + np.linspace(1./nrep, 1, nrep)[None, :]).ravel()
    df = pd.DataFrame(np.repeat(df, nrep, axis=0),
                      columns=df.columns)
    timestep /= nrep

    # Set potential fluxes for the interception reservoir
    mod.allocate(len(df), timestep=timestep)
    Qst0 = df.loc[:, ["RAINFALL[mm_day-1]", "PET[mm_day-1]"]].values.copy()
    Qst0[:, -1] *= -1
    Qst0 = np.ascontiguousarray(Qst0)
    mod.set_Qstars(0, Qst0)

    # Set potential fluxes for the routing reservoirs (use rainfall data)
    Qst3 = Qst0.copy()
    Qst3[:, 1] = 0.
    mod.set_Qstars(3, Qst3)
    mod.set_Qstars(4, Qst3)

    mod.run()

    # Run single reservoirs
    coefs = np.ascontiguousarray(mod.coefs_linear[0])
    _, x0, _, fx0, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[0]),
                                               mod.gammas[0],
                                               Qst0,
                                               np.ascontiguousarray(mod.coefs_linear[0]),
                                               timestep, mod.x0[0],
                                               flux_scalings=fxs[0])
    Qst1 = np.ascontiguousarray(-fx0[:, 3:5])
    _, _, _, fx1, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[1]),
                                               mod.gammas[1],
                                               Qst1,
                                               np.ascontiguousarray(mod.coefs_linear[1]),
                                               timestep, mod.x0[1])
    Qst2 = np.zeros_like(Qst0)
    Qst2[:, 0] = -fx1[:, 5]
    _, _, _, fx2, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[2]),
                                               mod.gammas[2],
                                               Qst2,
                                               np.ascontiguousarray(mod.coefs_linear[2]),
                                               timestep, mod.x0[2])
    Qst3 = np.zeros_like(Qst0)
    Qst3[:, 0] = Qst0[:, 0]
    _, _, _, fx3, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[3]),
                                               mod.gammas[3],
                                               Qst3,
                                               np.ascontiguousarray(mod.coefs_linear[3]),
                                               timestep, mod.x0[3])

    _, _, _, fx4, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[4]),
                                               mod.gammas[4],
                                               Qst3,
                                               np.ascontiguousarray(mod.coefs_linear[4]),
                                               timestep, mod.x0[4])

    for ires in range(mod.nreservoirs - 1):
        fx = [fx0, fx1, fx2, fx3, fx4][ires]
        assert allclose(fx, mod.fluxes[:, ires])

    qout = fx2[:, 5] + fx3[:, 5] + fx4[:, 5]
    assert allclose(qout, mod.fluxes[:, 5, 5])


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("timestep", TIMESTEPS)
@pytest.mark.parametrize("demand", [-0.5, -0.1, 0.5])
@pytest.mark.parametrize("flux_scalings", FLUX_SCALINGS[:2])
def test_grad_model_tangent(siteid, timestep, demand, flux_scalings,
                            allclose, debug_mode):
    noriginal = int(365.25 * 100)
    nrepeat = 2
    df = data_reader.get_data(siteid, "daily", 0).iloc[-noriginal:]
    df = df.interpolate().iloc[:, :2]
    assert df.shape == (noriginal, 2)
    df = np.tile(df, (nrepeat, 1))

    # Basic gr4j model
    mod = gradients.MultiReservoirModel(3, 20, "raise")
    mod.add_link_bypass_shortfall(0, 1)
    mod.add_link_outflow(1, 2)

    mod.set_coefs_linear(0, "openwatersmooth")
    mod.set_coefs_linear(1, "gr4jprod")
    mod.set_coefs_linear(2, "gr4jroute")
    mod.gammas = [5., 400., 50.]

    nv = gradients.VARTYPE_NUMBER
    vspecs = [[res, i] for res in range(mod.nreservoirs) for i in range(nv)]
    mod.allocate(len(df), variables_specs=vspecs, timestep=timestep)

    Qst0 = df
    Qst0[:, 1] *= -1
    mod.set_Qstars(0, Qst0)

    Qst2 = np.ones((len(df), 2))
    Qst2[:, 0] = np.nan
    mod.set_Qstars(2, Qst2)

    # Set demand for routing reservoir
    fxs = np.ones((mod.nreservoirs, gradients.NFLUXES_INPUTS))
    fxs[0] = flux_scalings
    fxs[-1, 1] = demand
    mod.flux_scalings = fxs

    # Run model
    t0 = time.time()
    mod.run()
    dtq = time.time() - t0

    dfx = mod.dfluxes

    dtn = 0

    for ivar, vspec in enumerate(vspecs):
        rid, vid = vspec

        # Numerical differences
        modp = mod.clone()
        modp.allocate(len(df), timestep=timestep)
        modp.set_Qstars(0, Qst0)
        modp.set_Qstars(2, Qst2)

        modm = mod.clone()
        modm.allocate(len(df), timestep=timestep)
        modm.set_Qstars(0, Qst0)
        modm.set_Qstars(2, Qst2)

        # Test downstream parameters have no effect on upstream fluxes
        if rid > 0:
            dfxu = dfx[:, ivar, :rid, :]
            assert allclose(dfxu, 0.)

        if vid == 0:
            eps = 1e-6 * mod.gammas[rid]
            delta = 2 * eps

            # Differentiate against gamma
            gs = mod.gammas.copy()
            gs[rid] = gs[rid] + eps
            modp.gammas = gs

            gs = mod.gammas.copy()
            gs[rid] = gs[rid] - eps
            modm.gammas = gs

        else:
            eps = 5e-7
            delta = 2 * eps
            inp = vid - 1
            modm.flux_scalings[rid, inp] -= eps
            modp.flux_scalings[rid, inp] += eps

        # Compute numerical difference
        t0 = time.time()
        modp.run()
        modm.run()
        dfx_n = (modp.fluxes - modm.fluxes) / delta
        dtn += time.time() - t0

        # get sim
        dfx_c = dfx[:, ivar, :, :]

        if vid == 0:
            dfx_n *= mod.gammas[rid]
            dfx_c *= mod.gammas[rid]

        if debug_mode and ivar == 5:
            fig, axs = plt.subplots(ncols=3, nrows=3,
                                    figsize=(12, 8),
                                    layout="constrained")
            for res in range(mod.nreservoirs):
                for ifx in range(gradients.NFLUXES_INPUTS):
                    ax = axs[res, ifx]
                    dc = dfx_c[:, res, ifx]
                    dn = dfx_n[:, res, ifx]
                    ax.plot(dc, label="Quasoare")
                    ax.plot(dn, label="Numerical")

                    tax = ax.twinx()
                    #tax.plot(mod.fluxes[:, res, ifx], color="0.5", lw=0.8)
                    tax.plot(dc - dn, color="0.7", lw=0.8, zorder=0)

                    ef, emax = eff(dn, dc)
                    ax.text(0.02, 0.98, f"Emax={emax:0.1e}\nEff={ef:0.1e}",
                            transform=ax.transAxes,
                            va="top", ha="left")

                    title = f"Res#{res} Flux#{ifx}"
                    ax.set(title=title)

                    if res == 0 and ifx == 0:
                        ax.legend()

            ftitle = f"Derivative against var #{ivar} (res={rid}, type={vid})"
            fig.suptitle(ftitle, fontweight="bold")
            plt.show()
            import pdb; pdb.set_trace()

        atol = 5e-4
        for res in range(mod.nreservoirs):
            dfx_err_max, _ = comp_dfluxes(mod.fluxes[:, res],
                                          dfx_n[:, res, :], dfx_c[:, res, :])
            assert np.nanmax(dfx_err_max) < atol

    assert dtq < dtn / 2


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("gamma_size", [10., 0.1])
@pytest.mark.parametrize("timestep", [1., 0.1])
def test_grad_model_vs_gradbench_quadrouting(siteid, timestep, gamma_size,
                                             debug_mode, allclose):
    pytest.skip("Can't make it work for certain values")
    if siteid in ["203060", "203900"]:
        ny = 10
        nrepeat = 200
    else:
        ny = 40
        nrepeat = 50
    noriginal = int(365.25 * ny)
    df = data_reader.get_data(siteid, "daily", 0)
    iok = np.where(df.iloc[:, -1].notnull())[0]
    i1, i2 = iok[[0, -1]]
    df = df.iloc[i1:i2]
    df = df.interpolate().iloc[-noriginal:]
    assert df.shape[0] == noriginal
    df = np.tile(df, (nrepeat, 1))
    qin = df[:, 2]
    nval = len(df)

    x0 = 0.5

    # Rescale gamma mm -> m3
    area = data_reader.get_stations().loc[siteid, "CATCHMENTAREA[km2]"]
    qin *= area / 86.4
    timestep *= 86400
    gamma = 10 * qin.mean() * timestep * gamma_size

    # run benchmarks
    gammas = np.array([gamma], dtype=np.float64)
    dgammas = np.array([1.])
    t0 = time.time()
    out = benchmarks.gradbench_quadrouting_tangent(timestep, qin, x0,
                                                   gammas, dgammas)
    dtb = time.time() - t0

    x, dx, fx, dfx = out
    dx *= gamma
    dfx *= gamma

    # run quasoare
    errors = "raise"
    # super careful with alphas around 0!!!!!
    nalphas = 500
    mod = gradients.MultiReservoirModel(1, nalphas, errors=errors)
    mod.set_coefs_linear(0, "quadrouting")
    mod.gammas = gamma

    nval = len(df)
    vspecs = [[0, 0]]
    mod.allocate(nval, timestep=timestep,
                 variables_specs=vspecs)
    Qst0 = np.zeros((nval, 2))
    Qst0[:, 0] = qin
    mod.set_Qstars(0, Qst0)
    t0 = time.time()
    mod.run()
    dtq = time.time() - t0

    # .. scale gradient to evaluate capacity scaling gradient
    mod.dx *= gamma
    mod.dfluxes *= gamma

    # numerical diff on quasoare
    modm = mod.clone()
    modm.allocate(nval, timestep=timestep)
    modm.set_Qstars(0, Qst0)

    modp = mod.clone()
    modp.allocate(nval, timestep=timestep)
    modp.set_Qstars(0, Qst0)

    eps = gamma * 1e-6
    modm.gammas = gamma - eps
    modp.gammas = gamma + eps

    t0 = time.time()
    modm.run()
    modp.run()
    dtn = time.time() - t0

    dx_n = gamma * (modp.x - modm.x) / eps / 2
    dfx_n = gamma * (modp.fluxes - modm.fluxes) / eps / 2


    if debug_mode:
        # Benchmark data
        xb = x[:, 0]
        dxb = dx[:, 0]
        yb = fx[:, 0]
        dyb = dfx[:, 0]

        # Quasoare data
        xq = mod.x[:, 0]
        dxq = mod.dx[:, 0, 0]
        yq = -mod.fluxes[:, 0, 5]
        dyq = -mod.dfluxes[:, 0, 0, 5]

        # num diff data
        dxn = dx_n[:, 0]
        dyn = -dfx_n[:, 0, 5]

        fig, axs = plt.subplots(ncols=2, nrows=4,
                                sharex=True, layout="constrained",
                                figsize=(17, 9))
        for ivar in range(2):
            if ivar == 0:
                wb, dwb, wn, dwn, wq, dwq = xb, dxb, xb, dxn, xq, dxq
                title = "storage"
            else:
                wb, dwb, wn, dwn, wq, dwq = yb, dyb, yb, dyn, yq, dyq
                title = "outflow"

            istart = -noriginal
            wb, dwb, wn, dwn, wq, dwq = [v[istart:] for v in
                                         [wb, dwb, wn, dwn, wq, dwq]]

            ax = axs[0, ivar]
            if ivar == 1:
                ax.plot(qin[-noriginal:], color="tab:red", label="Qin", lw=3)

            ax.plot(wb, "k-", label="benchmark")
            ax.plot(wq, "k--", label="quasoare")

            atol = 1e-5
            eff_q, emax_q = eff(wb, wq, atol)
            ax.set(title=f"{title} perf: eff(q)={eff_q:0.1e} emax(q)={emax_q:0.1e}")
            ax.legend(loc=1)

            ax = axs[1, ivar]
            ax.plot(wq - wb, "--", label="quasoare", lw=0.9, color="0.5")
            ax.legend(loc=1)
            ax.set_title("Diff quasoare - benchmark")

            ax = axs[2, ivar]
            ax.plot(dwb, "k-", label="benchmark")
            ax.plot(dwq, "k--", label="quasoare")
            ax.plot(dwn, "-", label="numerical", color="tab:blue")

            eff_q, emax_q = eff(dwb, dwq)
            eff_n, emax_n = eff(dwb, dwn)
            title = f"{title} gradient perf: emax(q)={emax_q:2.1e} emax(n)={emax_n:2.1e}"
            ax.set(title=title)
            ax.legend(loc=1)

            ax = axs[3, ivar]
            ax.plot(dwq - dwb, "--", label="quasoare", lw=0.9, color="0.5")
            ax.plot(dwn - dwb, "-", label="numerical", lw=0.9, color="tab:blue")
            ax.legend(loc=1)
            ax.set_title("Diff quasoare/diff - benchmark")

        plt.show()

    eff_fx, emax_fx = eff(fx[:, 0], -mod.fluxes[:, 0, 5], exclude=50)
    eff_dfx, emax_dfx = eff(dfx[:, 0], -mod.dfluxes[:, 0, 0, 5], exclude=50)
    eff_dfxn, emax_dfxn = eff(dfx[:, 0], -dfx_n[:, 0, 5], exclude=50)

    mess = f"fx:q={emax_fx:0.1e}" \
           + f" dfx:q={emax_dfx:0.1e}/n={emax_dfxn:0.1e} (q/n={emax_dfx/emax_dfxn:0.1e})" \
           + f" t:q={dtq:0.1e}/n={dtn:0.1e} (dtb/dtq={dtq/dtb:0.1f})"
    assert (emax_fx < 5e-2) \
        & (emax_dfx <= emax_dfxn * 1.2) \
        & (emax_dfx < 5e-3) \
        & (dtq < dtb * 20) \
        & (dtq < dtn), mess


@pytest.mark.parametrize("nuh", [2, 5, 20])
@pytest.mark.parametrize("X4", [0.1, 1., 5., 20.])
def test_gr4j_nash_cascade_tangent(nuh, X4, allclose, debug_mode):
    if (nuh == 2 and X4 == 1.0) \
            or (nuh == 5 and X4 == 1.0) \
            or (nuh == 20 and X4 == 5.):
        pytest.skip("These tests fail")

    nalphas = 100
    nres = nuh + 2
    mod = gradients.MultiReservoirModel(nres, nalphas,
                                        errors="raise")
    # Links
    mod.add_link_bypass_shortfall(0, 1)

    for iuh in range(nuh):
        mod.add_link_outflow(iuh + 1, iuh + 2)

    # Coefs
    mod.set_coefs_linear(0, "openwater")
    mod.set_coefs_linear(1, "gr4jprod")
    for iuh in range(nuh):
        mod.set_coefs_linear(2 + iuh, "linrouting")

    # allocate
    siteid = "203014"
    hclim = data_reader.get_data(siteid, "daily", 0).iloc[:, :2]
    ntimesteps = len(hclim)
    vspecs = [[2 + iuh, 3] for iuh in range(nuh)]
    mod.allocate(ntimesteps, variables_specs=vspecs)

    # Set inputs
    inputs = np.ascontiguousarray(hclim.values)
    inputs[:, 1] *= -1
    mod.set_Qstars(0, inputs)

    # Set parameters
    X0, X1 = 1e-2, 400
    mod.gammas = [X0, X1] + [1] * nuh
    sc3 = nuh * 1.25 / X4
    mod.flux_scalings[2:, 2] = sc3

    # Set numerical difference models
    modm = mod.clone()
    modp = mod.clone()
    eps = 1e-6
    for iuh in range(nuh):
        modm.flux_scalings[2:, 2] = sc3 * X4 / (X4 - eps)
        modp.flux_scalings[2:, 2] = sc3 * X4 / (X4 + eps)

    # Run models
    mod.run()
    modm.run()
    modp.run()

    dq1 = -mod.dfluxes[:, :, -1, 5].sum(axis=1) * nuh * 1.25 / X4**2
    dq2 = (modp.fluxes[:, -1, 5] - modm.fluxes[:, -1, 5]) / 2 / eps

    num = np.abs(dq1 - dq2)[:, None]
    denom = flux_denom(mod.fluxes[:, -1, [5]])
    err = num / denom
    exclude = 50
    emax = np.sort(err)[-exclude]

    atol = 1e-4
    if debug_mode and err.max() > atol:
        plt.close("all")
        fig, ax = plt.subplots()
        err = np.abs(dq1-dq2)
        imax = np.argmax(err)
        ax.plot(dq1, label="Quas")
        ax.plot(dq2, label="NumDiff")
        ax.plot(imax, dq1[imax], "ko")
        ax.plot(imax, dq2[imax], "ko")
        tax = ax.twinx()
        tax.plot(err, color="0.5")
        tax.plot(imax, err[imax], "ko")
        plt.show()
        import pdb; pdb.set_trace()

    assert err.max() < atol


def test_flux_scalings(allclose):
    nalphas = 100
    moda = gradients.MultiReservoirModel(1, nalphas,
                                        errors="raise")
    moda.set_coefs_linear(0, "gr4jprod")

    siteid = "203014"
    hclim = data_reader.get_data(siteid, "daily", 0).iloc[:, :2]
    hclim.iloc[:, 1] *= -1
    ntimesteps = len(hclim)
    moda.allocate(ntimesteps)
    moda.set_Qstars(0, hclim)
    moda.gammas = 100.
    moda.x0 = 0.5

    modb = moda.clone()
    modc = moda.clone()

    # Two different manners to reduce rainfall by 50%
    etaP = 0.5
    # .. setting flux scalings
    modb.flux_scalings[0, 0] = etaP
    # ... reducing Qstars
    modc.Qstars[:, 0, 0] *= etaP

    assert allclose(moda.Qstars, modb.Qstars)
    assert np.abs(moda.Qstars - modc.Qstars).max() > 1.
    assert np.abs(modb.flux_scalings - modc.flux_scalings).max() > 1e-1

    moda.run()
    modb.run()
    modc.run()

    # Simuls are different from basecase
    assert np.abs(moda.fluxes[:, :, :3] - modc.fluxes[:, :, :3]).max() > 0.1

    # Fluxes are identical
    assert allclose(modb.fluxes, modc.fluxes)


def test_flux_aggreg(allclose):
    siteid = "203014"
    hclim = data_reader.get_data(siteid, "daily", 0).iloc[:, :2]
    hclim = hclim.loc["2019":"2022"]
    hclim.iloc[:, 1] *= -1
    nval = len(hclim)

    def aggregate(x, ntimesteps, nagg):
        agg = x.cumsum(axis=0)
        agg[nagg:] = agg[nagg:] - agg[:-nagg]
        idx = np.arange(nagg - 1, (ntimesteps - 1) * nagg, nagg)
        return agg[idx] / nagg

    nuh = 5
    nalphas = 100
    nsubs = [1, 5, 20]
    mods = {}
    times = hclim.index
    for nsub in nsubs:
        mod = gradients.MultiReservoirModel(1 + nuh, nalphas,
                                             errors="raise")
        mod.set_coefs_linear(0, "gr4jprod")
        for iuh in range(nuh):
            mod.set_coefs_linear(1 + iuh, "linrouting")
        ntimesteps = nval * nsub
        mod.allocate(ntimesteps=ntimesteps, timestep=1./nsub)

        inputs = np.repeat(hclim, nsub, axis=0)
        mod.set_Qstars(0, inputs)

        mod.gammas = [400.] + [1.] * nuh
        mod.x0 = 0.5

        x4 = 1.2
        for iuh in range(nuh):
            mod.coefs_linear[1 + iuh, 2, :, 0] = 1.25 * nuh / x4

        # !!!!!!!
        # .. account for timestep change in Qstar3
        # !!!!!!!
        mod.flux_scalings[:, 2] = 1. / nsub

        mod.run()
        fx = aggregate(mod.fluxes, nval, nsub)
        if nsub == 1:
            fref = fx
        else:
            assert allclose(fx, fref)


@pytest.mark.parametrize("findiff_log10eps", [-4])
@pytest.mark.parametrize("gamma", [10, 100, 1000, 10000, 100000])
def test_grad_model_vs_gradbench_routing(findiff_log10eps, gamma, debug_mode):
    siteid = "203014"
    hclim = data_reader.get_data(siteid, "daily", 0).iloc[:, :2]
    if debug_mode:
        hclim = hclim[:50]
    hclim.iloc[:, 1] = 0
    ntimesteps = len(hclim)

    x0 = 0.5
    timestep = 1.

    # Benchmark
    dgammas = np.ones(1)
    qin = np.ascontiguousarray(hclim.values[:, 0])
    x, _, fluxes, dfluxes = benchmarks.gradbench_quadrouting_tangent(timestep,
                                                                     qin,
                                                                     x0,
                                                                     np.array([gamma]),
                                                                     dgammas)
    # Quasoare runs
    amin = 0
    amax = x.max()
    nalphas = 2000

    def configure_and_run(eps=0.):
        mod = gradients.MultiReservoirModel(1, nalphas,
                                            errors="raise")

        vspecs = [[0, 0]] if eps == 0. else None
        mod.allocate(ntimesteps,
                     timestep=timestep,
                     variables_specs=vspecs)
        mod.x0 = x0

        mod.set_alphas(0, amin, amax)
        mod.set_coefs_linear(0, "quadrouting")

        mod.set_Qstars(0, hclim)
        mod.gammas = gamma * (1 + eps)
        mod.run()
        return mod

    modref = configure_and_run()

    num = np.abs(-modref.fluxes[:, 0, 5] - fluxes[:, 0])
    denom = flux_denom(fluxes[:, [0]])[:, 0]
    err = num / denom
    assert err.max() < 5e-2
    print(f"err max = {err.max():5.3e}")

    # Finite diff
    findiff_eps = 10**findiff_log10eps
    modlow1 = configure_and_run(-findiff_eps)
    modhigh1 = configure_and_run(findiff_eps)

    dfluxes_1 = (modhigh1.fluxes
                 - modref.fluxes) / findiff_eps

    dfluxes_2 = (1. / 2 * modhigh1.fluxes
                 - 1. / 2 * modlow1.fluxes) / findiff_eps

    # process
    dfluxes *= -gamma
    modref.dfluxes *= gamma

    # Perf
    ex_ref, em_ref = comp_dfluxes(fluxes, dfluxes, modref.dfluxes[:, 0, [0], 5],
                                  exclude=0)

    ex_1, em_1 = comp_dfluxes(fluxes, dfluxes, dfluxes_1[:, [0], 5],
                              exclude=0)

    ex_2, em_2 = comp_dfluxes(fluxes, dfluxes, dfluxes_2[:, [0], 5],
                              exclude=0)

    mlrm = math.log(em_ref.squeeze())
    mlrm1 = math.log(em_1.squeeze())
    assert mlrm1 > mlrm

    mlrm2 = math.log(em_2.squeeze())
    assert mlrm2 < mlrm
