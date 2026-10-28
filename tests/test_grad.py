from pathlib import Path
import warnings
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
from pygme.models import gr2m

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
TIMESTEPS_AGG = [1., 5.]
GAMMAS = [10, 100, 10000]
FLUX_SCALINGS = [
    [1., 1., 1.],
    [1.2, 1., 1.],
    [1., 1.2, 1.],
    [1., 1., 1.2]
    ]
NUMDIFF_EPS = 1e-5

SITEIDS = ["203014", "203004"]

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
    netw = gradients.GradNetwork(nreservoirs)
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

def test_isabove():
    x = np.random.uniform(0, 1, size=(100, 3, 4, 5))
    isa = gradients.isabove(x, 50)

    q = np.percentile(x, 50, axis=0)[None, :]
    isa = np.where(~np.isnan(isa))
    expected = np.where(x > q)
    for a, b in zip(isa, expected):
        assert np.allclose(a, b)

def test_compare_denom(allclose):
    x = np.random.uniform(-1, 1, size=(100, 3, 4, 5))
    s = np.exp(np.random.uniform(-10, 1, size=x.shape[1:]))
    x = x * s[None, :]

    eps = 1e-2
    ax = np.abs(x)

    with pytest.raises(ValueError, match="Denominator data"):
        d = gradients.compare_denom(x, eps=eps)

    d = gradients.compare_denom(ax, eps=eps)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dm = np.nanmean(d, axis=0)
    axm = ax.mean(axis=0)
    assert (axm > eps).sum() > 0
    assert all(np.isnan(dm[axm < eps]))
    assert all(~np.isnan(dm[axm > eps]))

    isnan = np.where(axm[None, :] > eps, 0., np.nan)
    #expected = ax + eps1 + isnan
    expected = eps + ax.std(axis=0)[None, :] + isnan
    assert allclose(d, expected, equal_nan=True)


def test_compare(allclose):
    x1 = np.random.uniform(-1, 1, size=(100, 3, 4, 5))
    s = np.exp(np.random.uniform(-10, 1, size=x1.shape[1:]))
    x1 = x1 * s[None, :]

    x2 = np.random.uniform(0, 1, size=(100, 3, 4, 5))
    x2 = x2 * s[None, :]

    rp = [0, 50]
    eps = 1e-3
    em, st = gradients.compare(x1, x2, report_probs=rp,
                               eps=eps)

    ax1 = np.abs(x1)
    den = gradients.compare_denom(ax1, eps)
    err = np.abs(x1 - x2) / den

    for prob in rp:
        assert em[prob] == np.nanmax(st[prob])

        abov = gradients.isabove(ax1, prob)
        erra = err * abov
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            erram = np.nanmean(erra, axis=0)
        erram[np.isnan(erram)] = 0.
        assert allclose(st[prob], erram, equal_nan=True)


def test_grad_aggregate_fluxes(allclose):
    ntimesteps_agg = 100
    nsubsteps = 24
    ntimesteps = nsubsteps * ntimesteps_agg
    nres = 20
    nfxo = gradients.NFLUXES_OUTPUTS

    fx = np.random.uniform(-1, 1, size=(ntimesteps, nres, nfxo))
    fx_agg = np.empty((ntimesteps_agg - 1, nres, nfxo))
    with pytest.raises(ValueError, match="Number of timesteps"):
        ierr = c_pyquasoare_grad.grad_aggregate_fluxes(nsubsteps, fx, fx_agg)

    fx_agg = np.empty((ntimesteps_agg, nres, nfxo - 1))
    with pytest.raises(ValueError, match="fluxes_agg.shape"):
        ierr = c_pyquasoare_grad.grad_aggregate_fluxes(nsubsteps, fx, fx_agg)

    fx = np.random.uniform(-1, 1, size=(ntimesteps, nres, nfxo - 1))
    fx_agg = np.empty((ntimesteps_agg, nres, nfxo))
    with pytest.raises(ValueError, match="fluxes.shape"):
        ierr = c_pyquasoare_grad.grad_aggregate_fluxes(nsubsteps, fx, fx_agg)

    fx = np.random.uniform(-1, 1, size=(ntimesteps, nres, nfxo))
    fx_agg = np.empty((ntimesteps_agg, nres, nfxo))
    ierr = c_pyquasoare_grad.grad_aggregate_fluxes(nsubsteps, fx, fx_agg)

    for t in range(ntimesteps_agg):
        agg  = fx_agg[t]
        idx = np.arange(t * nsubsteps, (t + 1) * nsubsteps)
        expected = fx[idx].mean(axis=0)
        allclose(agg, expected)


def test_grad_aggregate_dfluxes(allclose):
    ntimesteps_agg = 100
    nsubsteps = 24
    ntimesteps = nsubsteps * ntimesteps_agg
    nvar = 50
    nres = 20
    nfxo = gradients.NFLUXES_OUTPUTS

    dfx = np.random.uniform(-1, 1, size=(ntimesteps, nvar, nres, nfxo))
    dfx_agg = np.empty((ntimesteps_agg - 1, nvar, nres, nfxo))
    with pytest.raises(ValueError, match="Number of timesteps"):
        ierr = c_pyquasoare_grad.grad_aggregate_dfluxes(nsubsteps, dfx, dfx_agg)

    dfx_agg = np.empty((ntimesteps_agg, nvar, nres, nfxo - 1))
    with pytest.raises(ValueError, match="dfluxes_agg.shape"):
        ierr = c_pyquasoare_grad.grad_aggregate_dfluxes(nsubsteps, dfx, dfx_agg)

    dfx = np.random.uniform(-1, 1, size=(ntimesteps, nvar, nres, nfxo - 1))
    dfx_agg = np.empty((ntimesteps_agg, nvar, nres, nfxo))
    with pytest.raises(ValueError, match="dfluxes.shape"):
        ierr = c_pyquasoare_grad.grad_aggregate_dfluxes(nsubsteps, dfx, dfx_agg)

    dfx = np.random.uniform(-1, 1, size=(ntimesteps, nvar, nres, nfxo))
    dfx_agg = np.empty((ntimesteps_agg, nvar, nres, nfxo))
    ierr = c_pyquasoare_grad.grad_aggregate_dfluxes(nsubsteps, dfx, dfx_agg)

    for t in range(ntimesteps_agg):
        agg  = dfx_agg[t]
        idx = np.arange(t * nsubsteps, (t + 1) * nsubsteps)
        expected = dfx[idx].mean(axis=0)
        allclose(agg, expected)


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


@pytest.mark.parametrize("siteid", SITEIDS)
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
    netw = gradients.GradNetwork(3)

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
    netw = gradients.GradNetwork(3)
    netw.add_link_outflow(0, 1)
    netw.add_link_outflow(1, 2)
    with pytest.raises(ValueError, match="cyclical"):
        netw.add_link_outflow(2, 0)


def test_network_wrong_order():
    netw = gradients.GradNetwork(3)
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

    # .. no error due to zero flux scaling
    gamma = 100.
    vtid = np.array([1] * nvars, dtype=np.int32)
    fxs = np.zeros(gradients.NFLUXES_INPUTS)
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, posinf, alphas, timestep, gamma,
                                            ipm, ipmw, qst, dqst, fxs, coefs_lin,
                                            vrid, vtid, niter, x, dx, fx, dfx)
    assert ierr == 0


@pytest.mark.parametrize("siteid", SITEIDS)
@pytest.mark.parametrize("reservoir", FLUX_FUNCTIONS)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("timestep", TIMESTEPS_AGG)
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
    timestep_agg = 1.
    siteid = "203004"
    df = data_reader.get_data(siteid, "daily")
    #df = df.iloc[:1]

    mod = gradients.MultiReservoirModel(2, errors="ignore")
    mod.set_coefs_linear(0, "openwater")
    mod.set_coefs_linear(1, "gr4jprod")
    mod.gammas = [5., 400.]
    mod.allocate(len(df), timestep_agg=timestep_agg)

    Qst0 = df.loc[:, ["RAINFALL[mm_day-1]", "PET[mm_day-1]"]].values.copy()
    Qst0[:, -1] *= -1
    Qst0 = np.ascontiguousarray(Qst0)

    coefs = np.ascontiguousarray(mod.coefs_linear[0])
    n0, x0, _, fx0, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[0]),
                                                 mod.gammas[0],
                                                 Qst0,
                                                 coefs,
                                                 timestep_agg, mod.x0[0])

    Qst1 = np.ascontiguousarray(-fx0[:, 3:5])
    n1, x1, _, fx1, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[1]),
                                                 mod.gammas[1],
                                                 Qst1,
                                                 np.ascontiguousarray(mod.coefs_linear[1]),
                                                 timestep_agg, mod.x0[1],
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


@pytest.mark.parametrize("siteid", SITEIDS)
@pytest.mark.parametrize("reservoir", FLUX_FUNCTIONS)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("timestep", TIMESTEPS_AGG)
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
    assert allclose(dSa, dSb, atol=5e-7)

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
    gamma = 1e-4
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
    assert allclose(outflow, expected, atol=5e-2)


@pytest.mark.parametrize("siteid", SITEIDS)
@pytest.mark.parametrize("reservoir", FLUX_FUNCTIONS)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("timestep", TIMESTEPS_AGG)
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

        x_em, x_st = gradients.compare(dx_n[:, None], dx_v[:, None])
        fx_em, fx_st = gradients.compare(dfx_n, dfx_v)

        xtol = 1e-3
        ftol = 8e-3

        if debug_mode and ((x_em > xtol).any() or (fx_em > ftol).any()):
            plt.close("all")
            ncols, nrows = 6, 2
            aw, ah = 3, 3
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                    figsize=(ncols * aw, nrows * ah),
                                    layout="constrained",
                                    sharex=True)
            den = gradients.compare_denom(np.abs(dfx_n))
            for ifx in range(6):
                ax = axs[0, ifx]
                ax.plot(fx[:, ifx], "k-")

                ax = axs[1, ifx]
                v = dfx_v[:, ifx]
                n = dfx_n[:, ifx]
                ax.plot(v, label="Quas")
                ax.plot(n, label="NumDiff")

                tax = ax.twinx()
                err = np.abs(v - n) / den[:, ifx]
                tax.plot(err, color="0.5", lw=0.8, alpha=0.5)

                title = f"Flux {ifx + 1}"
                ax.set_title(title)
                if ifx == 0:
                    ax.legend()

            ftitle = f"Diff against dvar={vid} eps={eps:0.1e}"
            fig.suptitle(ftitle, fontweight="bold")
            plt.show()

        #print(f"x={x_em.max():8.3e} fx={fx_em.max():8.3e}")
        assert x_em[50] < xtol
        assert fx_em[50] < ftol


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


@pytest.mark.parametrize("nsub", [1, 10])
def test_grad_model_clone(nsub, allclose):
    mod = gradients.MultiReservoirModel(2,
                                        nsubsteps=nsub,
                                        errors="raise")
    mod.add_link_bypass_shortfall(0, 1)
    mod.set_coefs_linear(0, "openwater")
    mod.set_coefs_linear(1, "gr4jprod")
    mod.gammas = [5., 400.]

    siteid = "203014"
    timestep_agg = 1.
    df = data_reader.get_data(siteid, "daily")
    mod.allocate(len(df), timestep_agg=timestep_agg,
                 variables_specs=[[0, 0], [1, 0]])
    Qst1 = df.loc[:, ["RAINFALL[mm_day-1]", "PET[mm_day-1]"]].values
    Qst1[:, -1] *= -1
    Qst1 = np.ascontiguousarray(Qst1)
    mod.set_Qstars_one_reservoir(0, Qst1)

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
        "niter",
        "nsubsteps",
        "ntimesteps_agg",
        "timestep",
        "timestep_agg",
        "fluxes",
        "fluxes_agg",
        "dfluxes",
        "dfluxes_agg"
        ]
    ntested = 0
    for vn in attrns:
        v1 = getattr(mod, vn)
        v2 = getattr(modc, vn)

        # .. check attribute is identical
        assert allclose(v1, v2, equal_nan=True)

        # .. perturb clone and check it becomes
        #    different from original
        try:
            v2.flat[0] += 1
        except:
            continue

        assert abs(v1.flat[0] - v2.flat[0]) > 0
        ntested += 1

    assert ntested == 13

@pytest.mark.parametrize("siteid", SITEIDS)
@pytest.mark.parametrize("timestep_agg", TIMESTEPS_AGG)
@pytest.mark.parametrize("flux_scalings", FLUX_SCALINGS)
@pytest.mark.parametrize("nsubsteps", [1, 2])
def test_grad_model(siteid, timestep_agg, flux_scalings, nsubsteps,
                    debug_mode, allclose):

    mod = gradients.MultiReservoirModel(5, nsubsteps=nsubsteps,
                                        errors="raise")

    # Model is the sum of 3 contributions summed via a transluent reservoir:
    # 1. Interception -> GR4Jprod -> GR4Jroute -> out
    # 2. GR4Jroute -> out
    # 3. GR4Jroute -> out
    mod.add_link_bypass_shortfall(0, 1)
    mod.add_link_outflow(1, 2)
    mod.add_link_outflow(2, 4)
    mod.add_link_outflow(3, 4)

    mod.set_coefs_linear(0, "openwater")
    mod.set_coefs_linear(1, "gr4jprod")
    mod.set_coefs_linear(2, "gr4jroute")
    mod.set_coefs_linear(3, "quadrouting")
    mod.set_coefs_linear(4, "transluent")
    mod.gammas = [5., 400., 50., 100., 1.]

    # Set flux scaling for the interception reservoir
    fxs = np.ones((mod.nreservoirs, gradients.NFLUXES_INPUTS))
    fxs[0] = flux_scalings
    mod.flux_scalings = fxs

    # Set potential fluxes for the interception reservoir
    df = data_reader.get_data(siteid, "daily")
    mod.allocate(len(df), timestep_agg=timestep_agg)
    Qst0 = df.loc[:, ["RAINFALL[mm_day-1]", "PET[mm_day-1]"]].values.copy()
    Qst0[:, -1] *= -1
    mod.set_Qstars_one_reservoir(0, Qst0)

    # Set potential fluxes for the routing reservoirs
    demand = -0.5
    Qst2 = np.zeros_like(Qst0)
    Qst2[:, 0] = np.nan
    Qst2[:, 1] = demand
    mod.set_Qstars_one_reservoir(2, Qst2)

    Qst3 = Qst0.copy()
    Qst3[:, 1] = 0.
    mod.set_Qstars_one_reservoir(3, Qst3)

    mod.run()

    # Run single reservoirs
    Qst0 = np.repeat(Qst0, nsubsteps, axis=0)
    timestep = mod.timestep
    fxs[:, 2] /= nsubsteps
    coefs = np.ascontiguousarray(mod.coefs_linear[0])
    _, x0, _, fx0, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[0]),
                                               mod.gammas[0],
                                               Qst0,
                                               np.ascontiguousarray(mod.coefs_linear[0]),
                                               timestep, mod.x0[0],
                                               flux_scalings=fxs[0])
    fxs = np.ones((mod.nreservoirs, gradients.NFLUXES_INPUTS))
    fxs[:, 2] /= nsubsteps
    Qst1 = np.ascontiguousarray(-fx0[:, 3:5])
    _, _, _, fx1, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[1]),
                                               mod.gammas[1],
                                               Qst1,
                                               np.ascontiguousarray(mod.coefs_linear[1]),
                                               timestep, mod.x0[1],
                                               flux_scalings=fxs[0])
    Qst2 = np.zeros((mod.ntimesteps, 2))
    Qst2[:, 0] = -fx1[:, 5]
    Qst2[:, 1] = demand
    _, _, _, fx2, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[2]),
                                               mod.gammas[2],
                                               Qst2,
                                               np.ascontiguousarray(mod.coefs_linear[2]),
                                               timestep, mod.x0[2],
                                               flux_scalings=fxs[0])
    Qst3 = np.zeros((mod.ntimesteps, 2))
    Qst3[:, 0] = Qst0[:, 0]
    _, _, _, fx3, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[3]),
                                               mod.gammas[3],
                                               Qst3,
                                               np.ascontiguousarray(mod.coefs_linear[3]),
                                               timestep, mod.x0[3],
                                               flux_scalings=fxs[0])

    # Individual reservoir outputs
    for ires in range(mod.nreservoirs - 1):
        fx = [fx0, fx1, fx2, fx3][ires]
        assert allclose(fx, mod.fluxes[:, ires])

    # Transluent output
    qout = fx2[:, 5] + fx3[:, 5]
    assert allclose(qout, mod.fluxes[:, -1, 5])


def test_grad_model_neg_fluxes():
    mod = gradients.MultiReservoirModel(1, errors="raise")

    # Flux functions are ok
    funs = [
        lambda x: np.maximum(x, 0),
        lambda x: np.maximum(x, 0),
        lambda x: np.maximum(x, 0),
        ]
    mod.set_coefs_linear(0, funs)
    mod.check_positive_fluxes()

    # f2 is negative -> problem
    funs = [
        lambda x: np.maximum(x, 0),
        lambda x: x - 0.5,
        lambda x: np.maximum(x, 0),
        ]
    mod.set_coefs_linear(0, funs)
    err = "Negative flux values in reservoir #0 - flux 1"
    with pytest.raises(ValueError, match=err):
        mod.check_positive_fluxes()

    # Should ok after fix
    mod.fix_negative_fluxes()
    mod.check_positive_fluxes()

    # Check we can run the model
    siteid = "203014"
    hclim = data_reader.get_data(siteid, "daily", 0).iloc[:, :2].values
    mod.gammas = 100
    mod.allocate(ntimesteps_agg=len(hclim))
    mod.x0 = 1.1
    Qst0 = hclim
    Qst0[:, 1] *= -1
    mod.set_Qstars_one_reservoir(0, Qst0)
    mod.run()


@pytest.mark.parametrize("siteid", SITEIDS)
@pytest.mark.parametrize("timestep_agg", TIMESTEPS_AGG)
@pytest.mark.parametrize("nsubsteps", [1, 4])
@pytest.mark.parametrize("demand", [-0.5, -0.1, 0.5])
@pytest.mark.parametrize("flux_scalings", FLUX_SCALINGS[:2])
def test_grad_model_tangent(siteid, timestep_agg, nsubsteps,
                            demand, flux_scalings,
                            allclose, debug_mode):
    if debug_mode:
        noriginal = 200
        nrepeat = 1
    else:
        noriginal = int(365.25 * 100)
        nrepeat = 2

    hclim = data_reader.get_data(siteid, "daily", 0).iloc[-noriginal:]
    hclim = hclim.interpolate().iloc[:, :2]
    assert hclim.shape == (noriginal, 2)
    hclim = np.tile(hclim, (nrepeat, 1))

    #hclim = hclim[:30]

    # Basic gr4j model
    mod = gradients.MultiReservoirModel(3,
                                        nalphas=50,
                                        nsubsteps=nsubsteps,
                                        errors="raise")
    mod.add_link_bypass_shortfall(0, 1)
    mod.add_link_outflow(1, 2)

    mod.set_coefs_linear(0, "openwater")
    mod.set_coefs_linear(1, "gr4jprod")
    mod.set_coefs_linear(2, "gr4jroute")
    mod.gammas = [1e-1, 400., 50.]

    nvs = np.arange(gradients.VARTYPE_NUMBER)
    #vspecs = [[res, i] for res in range(mod.nreservoirs) for i in nvs]
    vspecs = [[res, i] for res in range(1, mod.nreservoirs) for i in nvs]
    mod.allocate(ntimesteps_agg=len(hclim),
                 variables_specs=vspecs,
                 timestep_agg=timestep_agg)
    x0 = 0.5
    mod.x0 = x0
    Qst0 = hclim
    Qst0[:, 1] *= -1
    mod.set_Qstars_one_reservoir(0, Qst0)

    fxs = np.ones((mod.nreservoirs, gradients.NFLUXES_INPUTS))
    fxs[0] = flux_scalings
    if mod.nreservoirs > 2:
        Qst2 = np.ones((mod.ntimesteps_agg, 2))
        Qst2[:, 0] = np.nan
        mod.set_Qstars_one_reservoir(2, Qst2)

        # Set demand for routing reservoir
        fxs[2, 1] = demand

    mod.flux_scalings = fxs

    # Run model
    t0 = time.time()
    mod.run()
    mod.aggregate()
    dtq = time.time() - t0

    dfx = mod.dfluxes
    dfxa = mod.dfluxes_agg

    dtn = 0
    modp = mod.clone()
    modm = mod.clone()

    for ivar, vspec in enumerate(vspecs):
        rid, vid = vspec

        # Test downstream parameters have no effect on upstream fluxes
        if rid > 0:
            dfxu = dfx[:, ivar, :rid, :]
            assert allclose(dfxu, 0.)

        if vid == 0:
            e0 = 1e-7
            eps = e0 * max(1., mod.gammas[rid])
            delta = 2 * eps

            # Differentiate against gamma
            modp.gammas[rid] = mod.gammas[rid] + eps
            modm.gammas[rid] = mod.gammas[rid] - eps
        else:
            eps = 1e-7
            inp = vid - 1
            modm.flux_scalings[rid, inp] = mod.flux_scalings[rid, inp] - eps
            modp.flux_scalings[rid, inp] = mod.flux_scalings[rid, inp] + eps

            delta = 2 * eps
            if vid == 3:
                delta /= nsubsteps

        # Compute numerical difference
        t0 = time.time()
        modp.run()
        modp.aggregate()

        modm.run()
        modm.aggregate()

        dx_n = (modp.x - modm.x) / delta
        dfx_n = (modp.fluxes - modm.fluxes) / delta
        dfxa_n = (modp.fluxes_agg - modm.fluxes_agg) / delta

        dtn += time.time() - t0

        # Reset models
        if vid == 0:
            modp.gammas[rid] = mod.gammas[rid]
            modm.gammas[rid] = mod.gammas[rid]
        else:
            modm.flux_scalings[rid, inp] = mod.flux_scalings[rid, inp]
            modp.flux_scalings[rid, inp] = mod.flux_scalings[rid, inp]

        # get sim
        dfx_c = dfx[:, ivar, :, :]
        dfxa_c = dfxa[:, ivar, :, :]

        if debug_mode:
            plot_it = False
            atol = 5e-4
            emax = {}
            for res in range(mod.nreservoirs):
                fx = mod.fluxes_agg[:, res]
                em, st = gradients.compare(dfxa_n[:, res, :], dfxa_c[:, res, :])
                emax[res] = em[50]
                plot_it = em[50] > atol

            if not plot_it:
                continue

            plt.close("all")
            nres = mod.nreservoirs
            fig, axs = plt.subplots(ncols=4, nrows=nres,
                                    figsize=(15, 3 * nres),
                                    layout="constrained")
            axs = np.array(axs[None, :]) if nres == 1 else axs

            for res in range(mod.nreservoirs):
                ax = axs[res, 0]
                ax.plot(mod.dx[:, res, ivar], label="QuaSoARe")
                ax.plot(dx_n[:, res], label="FinDiff")
                tax = ax.twinx()
                tax.plot(np.abs(mod.dx[:, res, ivar] - dx_n[:, res]), color="0.5", lw=0.5)
                title = f"Res#{res} x"
                ax.set(title=title)

                denom = flux_denom(mod.fluxes[:, res])
                for ifx in range(gradients.NFLUXES_INPUTS):
                    ax = axs[res, 1 + ifx]

                    #dc = dfxa_c[:, res, ifx]
                    #dn = dfxa_n[:, res, ifx]
                    den = 1. + 0. * denom[:, ifx]
                    dc = dfx_c[:, res, ifx] / den
                    ax.plot(dc, label="QuaSoARe")

                    dn = dfx_n[:, res, ifx] / den
                    ax.plot(dn, label="FinDiff")

                    tax = ax.twinx()
                    tax.plot(np.abs(dc - dn), color="0.5", lw=0.5)

                    err = emax[res][ifx]
                    title = f"Res#{res} Flux#{ifx} err={err:0.2e}"
                    ax.set(title=title)

                    if res == 0 and ifx == 0:
                        ax.legend()

            ftitle = f"Derivative against var #{ivar} " \
                     + f" (nsub={nsubsteps} res={rid} type={vid})"
            fig.suptitle(ftitle, fontweight="bold")
            plt.show()
            import pdb; pdb.set_trace()

        # Perf
        em, _ = gradients.compare(dfx_n, dfx_c)
        atol = 1e-6
        errmsg = f"dfx em={em[50]:0.2e} > {atol:0.1e}"
        print(f"dfx em={em[50]:0.2e}")
        assert em[50] < atol, errmsg

    #if not debug_mode and min(dtq, dtn) > 0.:
    #    assert dtq < dtn * 0.9


@pytest.mark.parametrize("nuh", [1, 2, 4])
@pytest.mark.parametrize("nsubsteps", [2, 4])
@pytest.mark.parametrize("X1", [50, 400])
@pytest.mark.parametrize("X4", [0.1, 1., 5., 20.])
@pytest.mark.parametrize("varname", ["X1", "X4"])
def test_gr4j_nash_cascade_tangent(nuh, X1, X4, nsubsteps,
                                   varname, allclose, debug_mode):
    nalphas = 100
    nres = nuh + 2
    mod = gradients.MultiReservoirModel(nres, nalphas,
                                        nsubsteps=nsubsteps,
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
    nval = 5 * 365
    hclim = data_reader.get_data(siteid, "daily", 0).iloc[-nval:, :2]
    #hclim = hclim.iloc[:50]
    ntimesteps_agg = len(hclim)
    if varname == "X1":
        vspecs = [[1, 0]]
    else:
        vspecs = [[2 + iuh, 3] for iuh in range(nuh)]
    mod.allocate(ntimesteps_agg, variables_specs=vspecs)

    # Set inputs
    inputs = np.ascontiguousarray(hclim.values)
    inputs[:, 1] *= -1
    mod.set_Qstars_one_reservoir(0, inputs)

    # Set parameters
    X0 = 1e-2
    mod.gammas = [X0, X1] + [1] * nuh
    sc3 = nuh * 1.25 / X4
    mod.flux_scalings[2:, 2] = sc3

    # Set numerical difference models
    modm = mod.clone()
    modp = mod.clone()
    eps = 1e-7
    if varname == "X1":
        modm.gammas[1] = X1 - eps
        modp.gammas[1] = X1 + eps
    else:
        for iuh in range(nuh):
            modm.flux_scalings[2:, 2] = sc3 * X4 / (X4 - eps)
            modp.flux_scalings[2:, 2] = sc3 * X4 / (X4 + eps)

    # Run models
    mod.run()
    modm.run()
    modp.run()

    if varname == "X1":
        # Gradient against X1
        dq1 = mod.dfluxes[:, 0, :, :]
        dq2 = (modp.fluxes - modm.fluxes) / 2 / eps
        em, st = gradients.compare(dq1, dq2)
    else:
        # Gradient against X4 = dη/dX4 × Σdf/dη
        dq1 = -mod.dfluxes[:, :, -1, 5].sum(axis=1) * nuh * 1.25 / X4**2
        dq2 = (modp.fluxes[:, -1, 5] - modm.fluxes[:, -1, 5]) * mod.nsubsteps / 2 / eps
        dq1, dq2 = dq1[:, None, None], dq2[:, None, None]
        em, st = gradients.compare(dq1, dq2)

    atol = 1e-4
    if debug_mode and em.max() > atol:
        #imax = np.where(st[90] == st[90].max())
        imax = np.where(st[90] > 1e-1)
        res, ifx = [ii[0] for ii in imax]

        plt.close("all")
        fig, axs = plt.subplots(nrows=3, sharex=True, layout="constrained")
        ax = axs[0]
        y1 = mod.x[:, res]
        ax.plot(y1, label="Quas")
        ax.set(title="Store")

        ax = axs[1]
        y1 = mod.fluxes[:, res, ifx]
        y2 = modp.fluxes[:, res, ifx]
        ax.plot(y1, label="Quas")
        ax.plot(y2, label="NumDiff+")
        tax = ax.twinx()
        tax.plot(y2 - y1, color="0.5", lw=0.5)
        ax.set(title="Flux")

        ax = axs[2]
        y1 = dq1[:, res, ifx]
        y2 = dq2[:, res, ifx]
        ax.plot(y1, label="Quas")
        ax.plot(y2, label="NumDiff")
        tax = ax.twinx()
        den = gradients.compare_denom(np.abs(y1)[:, None])
        err = np.abs(y1-y2) / den[:, 0]
        tax.plot(err, color="0.5", lw=0.5)
        ax.set(title="grad")

        ftitle = f"Res {res} - Fx {ifx} / var={varname}"
        fig.suptitle(ftitle, fontweight="bold")
        plt.show()
        import pdb; pdb.set_trace()

    assert em[50] < atol


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
    moda.set_Qstars_one_reservoir(0, hclim)
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


@pytest.mark.parametrize("reservoir", ["gr4jprod", "quadrouting"])
@pytest.mark.parametrize("nsub", [1, 2, 6, 100])
def test_flux_aggreg(reservoir, nsub, allclose):
    siteid = "203014"
    hclim = data_reader.get_data(siteid, "daily", 0).iloc[:, :2]
    hclim.iloc[:, 1] *= -1
    hclim = hclim.loc["2022"]
    ntimesteps_agg = len(hclim)

    nalphas = 100
    mod = gradients.MultiReservoirModel(1, nalphas,
                                        nsubsteps=nsub,
                                        errors="raise")
    mod.set_coefs_linear(0, reservoir)
    mod.allocate(ntimesteps_agg=ntimesteps_agg,
                 variables_specs=[[0, 0], [0, 1]])
    mod.set_Qstars_one_reservoir(0, hclim)
    mod.gammas = [500.]
    mod.x0 = 0.5
    mod.run()
    mod.aggregate()

    t0 = hclim.index
    assert mod.ntimesteps_agg == len(t0)

    dt = pd.Timedelta(days=1./nsub)
    t1 = pd.date_range(t0[0], t0[-1] + dt * (nsub - 1), freq=dt)
    assert mod.ntimesteps == len(t1)

    for i in range(ntimesteps_agg):
        idx0 = t0 == t0[i]
        fx0 = mod.fluxes_agg[idx0]

        idx1 = (t1 >= t0[i]) & (t1 < t0[i] + pd.Timedelta(days=1))
        fx1 = mod.fluxes[idx1]
        assert allclose(fx1.mean(axis=0), fx0)

        dfx0 = mod.dfluxes_agg[idx0]
        dfx1 = mod.dfluxes[idx1]
        assert allclose(dfx1.mean(axis=0), dfx0)


def test_flux_aggreg_gr4jprod(allclose):
    siteid = "203014"
    hclim = data_reader.get_data(siteid, "daily", 0).iloc[:, :2]
    hclim = hclim.loc["1990":"2023"]
    hclim.iloc[:, 1] *= -1
    ntimesteps_agg = len(hclim)

    nuh = 5
    nalphas = 100
    nsubs = [1, 2, 10, 20]
    mods = {}
    times = hclim.index
    for nsub in nsubs:
        mod = gradients.MultiReservoirModel(1, nalphas,
                                            nsubsteps=nsub,
                                            errors="raise")
        mod.set_coefs_linear(0, "gr4jprod")
        mod.allocate(ntimesteps_agg=ntimesteps_agg)
        mod.set_Qstars_one_reservoir(0, hclim)
        mod.gammas = [400.]
        mod.x0 = 0.5

        mod.run()
        mod.aggregate()
        if nsub == 1:
            fref = mod.fluxes_agg.copy()
        else:
            assert allclose(mod.fluxes_agg, fref)


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
    timestep_agg = 1.

    # Benchmark
    dgammas = np.ones(1)
    qin = np.ascontiguousarray(hclim.values[:, 0])
    x, _, fluxes, dfluxes = benchmarks.gradbench_quadrouting_tangent(timestep_agg,
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
                     timestep_agg=timestep_agg,
                     variables_specs=vspecs)
        mod.x0 = x0

        mod.set_alphas(0, amin, amax)
        mod.set_coefs_linear(0, "quadrouting")

        mod.set_Qstars_one_reservoir(0, hclim)
        mod.gammas = gamma * (1 + eps)
        mod.run()
        mod.aggregate()
        return mod

    modref = configure_and_run()

    # Check flux performance
    em, _ = gradients.compare(fluxes[:, [0]],
                              -modref.fluxes_agg[:, 0, [5]])
    assert em.max() < 5e-3

    # Finite diff
    findiff_eps = 10**findiff_log10eps
    modlow1 = configure_and_run(-findiff_eps)
    modhigh1 = configure_and_run(findiff_eps)

    dfluxes_1 = (modhigh1.fluxes_agg
                 - modref.fluxes_agg) / findiff_eps

    dfluxes_2 = (1. / 2 * modhigh1.fluxes_agg
                 - 1. / 2 * modlow1.fluxes_agg) / findiff_eps

    # Perf
    g = -gamma
    em_ref, _ = gradients.compare(g * dfluxes,
                                  -g * modref.dfluxes_agg[:, 0, [0], 5])

    em_1, _ = gradients.compare(g * dfluxes,
                                dfluxes_1[:, [0], 5])

    em_2, _ = gradients.compare(g * dfluxes,
                                dfluxes_2[:, [0], 5])

    mlrm = math.log(em_ref.max())
    mlrm1 = math.log(em_1.max())
    assert mlrm1 > mlrm

    mlrm2 = math.log(em_2.max())
    assert mlrm2 < mlrm


@pytest.mark.parametrize("siteid", SITEIDS)
@pytest.mark.parametrize("timestep_agg", [1., 5., 0.1])
@pytest.mark.parametrize("nsubsteps", [2])
def test_openwater(siteid, timestep_agg, nsubsteps, allclose):
    hclim = data_reader.get_data(siteid, "daily", 0).iloc[:, :2]
    hclim.iloc[:, 1] *= -1
    ntimesteps = len(hclim)

    nalphas = 10
    mod = gradients.MultiReservoirModel(1, nalphas,
                                        nsubsteps,
                                        errors="raise")
    mod.allocate(ntimesteps,
                 timestep_agg=timestep_agg)
    mod.x0 = 0.5
    mod.set_coefs_linear(0, "openwater")

    mod.set_Qstars_one_reservoir(0, hclim)
    mod.gammas = [1e-4]
    mod.run()
    mod.aggregate()

    Pi = -mod.fluxes_agg[:, 0, 3]
    expected = np.maximum(hclim.sum(axis=1), 0).values
    atol = 1e-3
    assert allclose(Pi, expected, atol=atol)

    Ei = mod.fluxes_agg[:, 0, 4]
    expected = np.maximum(-hclim.sum(axis=1), 0)
    assert allclose(Ei, expected, atol=atol)


@pytest.mark.parametrize("siteid", SITEIDS)
@pytest.mark.parametrize("X1", [10, 100, 10000])
def test_gr2m_prod(siteid, X1, allclose):
    hclim = data_reader.get_data(siteid, "daily", 0).iloc[:, :2]
    hclim.iloc[:, 1] *= -1
    hclim = hclim.resample("MS").sum()
    ntimesteps = len(hclim)

    nalphas = 500
    mod = gradients.MultiReservoirModel(1, nalphas,
                                        nsubsteps=1,
                                        errors="raise")
    mod.allocate(ntimesteps)
    funs = gradients.FLUX_FUNCTIONS["gr2mprod"]
    mod.set_coefs_linear(0, funs)
    mod.set_Qstars_one_reservoir(0, hclim)
    mod.gammas = X1
    mod.x0 = 0.5
    mod.run()
    mod.aggregate()

    nval = len(hclim)
    S = X1 / 2
    out = []
    nsub = 1000
    delta = 1. / nsub
    for i in range(nval):
        P, E = hclim.iloc[i]

        # Substepping of original GR2M
        # (i.e. operator splitting)
        P1, AE, P2, P3 = 0, 0, 0, 0
        for k in range(nsub):
            PHI = math.tanh(delta * P / X1)
            S1 = (S + X1 * PHI) / (1 + PHI * S / X1)
            iP1 = P * delta + S - S1
            P1 += iP1

            PSI = math.tanh(delta * -E / X1)
            S2 = S1 * (1 - PSI) / (1 + PSI * (1 - S1 / X1))
            AE += S1 - S2

            Sr = S2 / X1
            S = S2 / math.cbrt(1 + delta * Sr*Sr*Sr)
            iP2 = S2 - S
            P2 += iP2
            P3 += iP1 + iP2

        out.append({"S": S, "AE": AE, "P1": P1, "P2":P2, "P3": P3})

    df = pd.DataFrame(out)
    if nsub == 1:
        gm0 = gr2m.GR2M()
        gm0.allocate(np.abs(hclim), noutputs=gm0.noutputsmax)
        gm0.X1 = X1
        gm0.initialise([X1 / 2, 30])
        gm0.run()
        df0 = gm0.to_dataframe()
        assert allclose(df, df0.loc[:, df.columns], atol=5e-3)

    assert allclose(mod.x[:, 0], df.S / X1, atol=5e-2)
    assert allclose(-mod.fluxes[:, 0, 5], df.P3, atol=5e-2)


@pytest.mark.parametrize("siteid", SITEIDS)
@pytest.mark.parametrize("timestep_agg", [1.])
@pytest.mark.parametrize("nsubsteps", [1, 5, 20])
def test_openwater_grad(siteid, timestep_agg, nsubsteps, allclose):
    # Exploring bug suggested by Francois Bourgin
    hclim = data_reader.get_data(siteid, "daily", 0).iloc[:, :2]
    hclim.iloc[:, 1] *= -1
    ntimesteps = len(hclim)

    nalphas = 10
    mod = gradients.MultiReservoirModel(1, nalphas,
                                        nsubsteps,
                                        errors="raise")
    vspecs = [[0, 0], [0, 1], [0, 2], [0, 3]]
    mod.allocate(ntimesteps,
                 timestep_agg=timestep_agg,
                 variables_specs=vspecs)
    mod.x0 = 0.5
    mod.set_coefs_linear(0, "openwater")

    mod.set_Qstars_one_reservoir(0, hclim)
    mod.gammas = [0.1]
    mod.run()

    nval = len(hclim)
    d = np.arange(nsubsteps)[None, :] / nsubsteps
    time = np.arange(nval)[:, None] + d
    time = time.ravel()

    # Finite diff
    modp = mod.clone()
    modm = mod.clone()
    eps = 1e-8
    for ivar in range(len(vspecs)):
        vtype = vspecs[ivar][1]
        # Reset
        modp.gammas = mod.gammas
        modm.gammas = mod.gammas
        modp.flux_scalings[:] = 1
        modm.flux_scalings[:] = 1

        if vtype == 0:
            modp.gammas = mod.gammas * (1 + eps)
            modm.gammas = mod.gammas * (1 - eps)
            delta = mod.gammas[0] * 2 * eps
        else:
            ifx = vtype - 1
            modp.flux_scalings[0, ifx] = mod.flux_scalings[0, ifx] * (1 + eps)
            modm.flux_scalings[0, ifx] = mod.flux_scalings[0, ifx] * (1 - eps)
            delta = mod.flux_scalings[0, ifx] * 2 * eps

        modp.run()
        modm.run()
        dfx = (modp.fluxes - modm.fluxes) / delta
        dfx_m = mod.dfluxes[:, ivar, :, :]


@pytest.mark.parametrize("nsubsteps", [1, 5, 20])
@pytest.mark.parametrize("ifun", [1, 2, 3])
def test_steady_grad(nsubsteps, ifun, allclose):
    nalphas = 8
    mod = gradients.MultiReservoirModel(1, nalphas,
                                        nsubsteps,
                                        errors="raise")
    vspecs = [[0, 0], [0, 1], [0, 2], [0, 3]]
    nval = 50
    mod.allocate(nval,
                 timestep_agg=1,
                 variables_specs=vspecs)
    mod.x0 = 0.6

    # Test different smoothing variables
    if ifun == 1:
        fun = lambda x: np.where(x > 0, 1., 0.)
    elif ifun == 2:
        fun = lambda x: x / np.sqrt(x**2 + 1e-4)
    else:
        b = 1e-2
        fun = lambda x: np.where(x < 0, 0,
                            np.where(x < b, x**2/2/b**2,
                                np.where(x < 2 * b, 1 - (x - 2*b)**2/2/b**2,
                                    1)))
    funs = [
        lambda x: fun(1 - x),
        lambda x: fun(x),
        lambda x: np.zeros_like(x)
        ]
    mod.set_coefs_linear(0, funs)

    inputs = np.ones((nval, 2))
    inputs[:, -1] *= -1.1
    mod.set_Qstars_one_reservoir(0, inputs)
    mod.gammas = [10]
    mod.run()

    # Finite diff
    modp = mod.clone()
    modm = mod.clone()
    eps = 1e-8
    for ivar in range(len(vspecs)):
        vtype = vspecs[ivar][1]

        # Reset
        modp.gammas = mod.gammas
        modm.gammas = mod.gammas
        modp.flux_scalings[:] = 1
        modm.flux_scalings[:] = 1

        if vtype == 0:
            delta = mod.gammas * eps
            modp.gammas = mod.gammas + delta
            modm.gammas = mod.gammas - delta
        else:
            ifx = vtype - 1
            delta = mod.flux_scalings[0, ifx] * eps
            modp.flux_scalings[0, ifx] = mod.flux_scalings[0, ifx]  + delta
            modm.flux_scalings[0, ifx] = mod.flux_scalings[0, ifx]  - delta

        modp.run()
        modm.run()
        dfx = (modp.fluxes[:, 0] - modm.fluxes[:, 0]) / delta / 2
        dfx_m = mod.dfluxes[:, ivar, 0]
        assert allclose(dfx, dfx_m, atol=5e-8)
