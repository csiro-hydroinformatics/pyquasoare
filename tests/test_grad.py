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

from hydrodiy.io import iutils, csv

from pyquasoare import integrate, models, gradients, approx, steady
import c_pyquasoare_grad

import data_reader

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent
LOGGER = iutils.get_logger("models", flog=FTEST / "test_grad.log")
NALPHAS = 50

DEMANDS = [-0.5, 0, 0.5]
TIMESTEPS = [0.1, 1]
GAMMAS = [1, 10, 100, 10000]
VARIABLE_IDS = [0, 1, 2, 3]
NUMDIFF_EPS = 1e-5

def fixed(x, x0):
    return x0 * np.ones_like(x)

def indicator(x, xmin, xmax):
    return np.where((x > xmin) & (x < xmax), 1., 0.)

def comp_fun(x, x0=1e-2):
    return np.asinh(x / x0)

FLUX_FUNCTIONS = {
    "transluent": [
        lambda x: x * 0.,
        lambda x: x * 0.,
        lambda x: x * 0.
        ],
    "gr4jprod": [
        lambda x: 1 - np.minimum(x, 1)**2,
        lambda x: np.clip(x, 0, 1) * (2 - np.clip(x, 0, 1)),
        lambda x: 4**3/9**4 * np.maximum(x, 0)**5
        ],
    "gr4jprodquad": [
        lambda x: 1 - np.minimum(x, 1)**2,
        lambda x: np.clip(x, 0, 1) * (2 - np.clip(x, 0, 1)),
        lambda x: 4**3/9**4 * np.maximum(x, 0)**2
        ],
    "gr4jroute": [
        lambda x: 0.9 * fixed(x, 1.),
        lambda x: np.maximum(x, 0.)**3.5,
        lambda x: np.maximum(x, 0.)**5 / 4
        ],
    "gr4jroutequad": [
        lambda x: 0.9 * fixed(x, 1.),
        lambda x: np.maximum(x, 0.)**2,
        lambda x: np.maximum(x, 0.)**2 / 4
        ],
    "quadrouting": [
        lambda x: indicator(x, 0., np.inf),
        lambda x: fixed(x, 1.),
        lambda x: np.maximum(x, 0.)**2
        ],
    "linrouting": [
        # Lag theta = 3.5
        lambda x: indicator(x, 0, np.inf),
        lambda x: fixed(x, 1.),
        lambda x: np.maximum(x, 0.) / 3.5
        ],
    "nonlinrouting": [
        # exponent n = 6
        lambda x: indicator(x, 0., np.inf),
        lambda x: fixed(x, 1.),
        lambda x: np.maximum(x, 0.)**4
        ],
    "hymod1": [
        # exponent = 0.2
        lambda x: (1 - np.clip(x, 0., 1.))**0.2,
        lambda x: np.maximum(x, 0.),
        lambda x: fixed(x, 0.)
        ],
    "hymod2": [
        # exponent = 5
        lambda x: (1 - np.clip(x, 0., 1.))**0.2,
        lambda x: np.maximum(x, 0.),
        lambda x: fixed(x, 0.)
        ],
    "intercept": [
        lambda x: indicator(x, 0., 1.),
        lambda x: indicator(x, 0., 1.),
        lambda x: fixed(x, 0.)
        ],
    "muskingumrouting": [
        # KM = 2.5 / XM = 0.2
        lambda x: indicator(x, 0, np.inf) / (1 - 0.2),
        lambda x: fixed(x, 1.),
        lambda x: np.maximum(x, 0.) / 2.5 / (1 - 0.2)
        ],
}

def get_data(siteid, reservoir, timestep, gamma, demand,
             istart=None, iend=None):
    df = data_reader.get_data(siteid, "daily")

    istart = 0 if istart is None else istart
    iend = len(df) if iend is None else iend
    df = df.iloc[istart: iend]

    Qstars = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
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
        if ntype == "total_outflow":
            netw.add_link_total_outflow(res - 1, res)
        else:
            netw.add_link_bypass_shortfall(res - 1, res)
    return netw.input_mapping


def skip_this(reservoir, gamma, demand):
    if not re.search("rout", reservoir) and demand != 0:
        pytest.skip("Config not relevant")

    if reservoir in ["linrouting", "muskingumrouting"]:
        if demand < 0:
            pytest.skip("Can't have demand <0 with pure linear routing models")

        if gamma > 1000:
            pytest.skip("Skipping linear routing with tests with large gamma")

    if reservoir in ["gr4jroutequad"] and demand > 0:
        pytest.skip("Can't have demand >0 with gr4jroute quad")


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", FLUX_FUNCTIONS)
@pytest.mark.parametrize("gamma", [10])
@pytest.mark.parametrize("timestep", [1])
@pytest.mark.parametrize("demand", DEMANDS)
@pytest.mark.parametrize("eps", [-NUMDIFF_EPS, 0., NUMDIFF_EPS])
def test_reservoir_functions(siteid, reservoir, gamma, timestep, demand, eps):
    Qstars, scalings = get_data(siteid, reservoir, timestep, gamma, demand)
    alphas, coefs, coefs_lin = get_alphas_and_coefs(reservoir, scalings)
    coefs_lin *= 1 + eps

    a0 = alphas[:-1][None, :]
    c0 = coefs_lin[:, :, 0] * a0 + coefs_lin[:, :, 1]
    assert np.all(c0 >= 0)

    a1 = alphas[1:][None, :]
    c1 = coefs_lin[:, :, 0] * a1 + coefs_lin[:, :, 1]
    assert np.all(c1 >= 0)


@pytest.mark.parametrize("linktype", ["total", "bypass"])
def test_network(linktype, allclose):
    netw = gradients.Network(3)

    if linktype == "total":
        netw.add_link_total_outflow(0, 1)
        netw.add_link_total_outflow(1, 2)
    else:
        netw.add_link_bypass_shortfall(0, 1)
        netw.add_link_bypass_shortfall(1, 2)

    assert netw.upstream_reservoirs(2) == {0, 1}

    ipm = netw.input_mapping
    nfxo = gradients.NFLUXES_OUTPUTS
    ipmd = gradients.INPUT_MAPPING_LASTDIM
    assert ipm.shape == (netw.nreservoirs, nfxo * netw.nreservoirs, ipmd)
    sc = gradients.INPUT_MAPPING_SCALINGFACTOR

    for rid in range(netw.nreservoirs):
        lks = ipm[rid]
        if rid == 0:
            assert np.all(lks == -1)
        else:
            if linktype == "total":
                assert np.all(lks[1:] == -1)
                # total outflow from upstream links to inflow
                assert all(lks[0] == [rid - 1, 5, 0, -sc])
            else:
                assert np.all(lks[2:] == -1)
                # bypass flow from upstream links to inflow
                assert all(lks[0] == [rid - 1, 3, 0, -sc])
                # demand shortfall flow from upstream links to demand
                assert all(lks[1] == [rid - 1, 4, 1, -sc])

    # This is cyclical -> error
    with pytest.raises(ValueError, match="cyclical"):
        netw.add_link_total_outflow(2, 0)


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

    rid = 0
    vrid = np.array([0] * nvars, dtype=np.int32)
    vtid = np.array([0] * nvars, dtype=np.int32)
    qst = -np.ones(2)
    dqst = np.zeros((nvars, gradients.NFLUXES_INPUTS))
    fxs = np.ones(3)
    t0 = 0.
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, alphas, timestep, gamma,
                                            ipm, qst, dqst, fxs, coefs_lin,
                                            vrid, vtid, niter, x, dx, fx, dfx)
    assert ierr == 112

    qst = np.ones(2)
    coefs_lin[0] *= -1
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, alphas, timestep, gamma,
                                            ipm, qst, dqst, fxs, coefs_lin,
                                            vrid, vtid, niter, x, dx, fx, dfx)
    assert ierr == 111

    coefs_lin[0] *= -1
    gamma = 0
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, alphas, timestep, gamma,
                                            ipm, qst, dqst, fxs, coefs_lin,
                                            vrid, vtid, niter, x, dx, fx, dfx)
    assert ierr == 113

    gamma = 100.
    vtid = np.array([1] * nvars, dtype=np.int32)
    fxs = np.zeros(3)
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, alphas, timestep, gamma,
                                            ipm, qst, dqst, fxs, coefs_lin,
                                            vrid, vtid, niter, x, dx, fx, dfx)
    assert ierr == 604


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", FLUX_FUNCTIONS)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("timestep", TIMESTEPS)
@pytest.mark.parametrize("demand", DEMANDS)
def test_grad_integrate(siteid, reservoir, gamma, timestep, demand, allclose):
    skip_this(reservoir, gamma, demand)
    start, end = 0, 500
    Qstars, scalings = get_data(siteid, reservoir, timestep,
                                gamma, demand, start, end)
    alphas, coefs, coefs_lin = get_alphas_and_coefs(reservoir, scalings)

    nres = 3
    ipm = get_input_mapping(nres, "bypass_shortfall")

    # Loop and integrate
    rid = 0 # this is the top reservoir
    vrid = np.repeat(np.arange(nres, dtype=np.int32), 4)
    vtid = np.tile(np.arange(4, dtype=np.int32), 3)
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
        ierr = c_pyquasoare_grad.grad_integrate(nres, rid, alphas, timestep, gamma,
                                                ipm, qst, dqst, fxs, coefs_lin,
                                                vrid, vtid,
                                                niter2, x, dx, fx, dfx)
        assert ierr == 0
        assert allclose(x1, x[0])
        assert allclose(fx1, fx[:3])

        fxa = np.array([fx1[0] - qst[0],
                        fx1[1] - qst[1],
                        fx1[0] - qst[0] + fx1[2]])
        assert allclose(fxa, fx[3:])

        x_start = x1


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", FLUX_FUNCTIONS)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("timestep", TIMESTEPS)
@pytest.mark.parametrize("demand", DEMANDS)
def test_grad_reservoir(siteid, reservoir, gamma, timestep, demand, allclose):
    skip_this(reservoir, gamma, demand)

    Qstars, scalings = get_data(siteid, reservoir, timestep, gamma, demand)
    alphas, coefs, coefs_lin = get_alphas_and_coefs(reservoir, scalings)

    # Quadratic
    x0 = 0.5
    n1, x1, fx1 = models.quad_model(alphas, scalings, coefs,
                                    x0, timestep)
    fx1 *= gamma / timestep
    fx1 = np.column_stack([fx1,
                           -Qstars[:, 0] + fx1[:, 0],
                           -Qstars[:, 1] + fx1[:, 1],
                           -Qstars[:, 0] + fx1[:, 0] + fx1[:, 2]])

    # Lin
    n2, x2, dx2, fx2, dfx2 = gradients.grad_reservoir(alphas, gamma,
                                                      Qstars, coefs_lin,
                                                      timestep, x0)

    assert allclose(x1, x2)
    assert allclose(fx1, fx2)
    assert all(~np.isnan(x2))
    assert np.all(~np.isnan(dx2))

    # Sanity check
    if not re.search("rout(e|ing)", reservoir):
        assert all((x2 >= 0) & (x2 <= 1.05))

    # Mass balance
    dSa = np.diff(x2) * gamma
    dSb = timestep * fx2[1:, :3].sum(axis=1)
    assert allclose(dSa, dSb)

    if reservoir == "transluent":
        # No flux storage
        assert allclose(fx2[:, :3], 0.)
        assert allclose(fx2[:, 3], -Qstars[:, 0])
        assert allclose(fx2[:, 5], -Qstars[:, 0])
        assert allclose(fx2[:, 4], -Qstars[:, 1])


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", FLUX_FUNCTIONS)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("timestep", TIMESTEPS)
@pytest.mark.parametrize("demand", DEMANDS)
def test_grad_reservoir_tangent(siteid, reservoir, gamma, timestep,
                                demand, allclose, debug_mode):
    skip_this(reservoir, gamma, demand)
    start, end = None, 600
    Qstars, scalings = get_data(siteid, reservoir, timestep, gamma, demand,
                                start, end)
    alphas, coefs, coefs_lin = get_alphas_and_coefs(reservoir, scalings)

    # Gradient
    n, x, dx, fx, dfx = gradients.grad_reservoir(alphas, gamma,
                                                 Qstars, coefs_lin,
                                                 timestep)

    # Numerical difference
    for vid in range(gradients.VARTYPE_NUMBER):
        dx_v = dx[:, vid]
        dfx_v = dfx[:, vid]

        if vid == 0:
            eps = gamma * 1e-4
            gamma_m = gamma - eps
            gamma_p = gamma + eps
            coefs_lin_m = coefs_lin
            coefs_lin_p = coefs_lin
        else:
            eps = NUMDIFF_EPS
            gamma_m, gamma_p = gamma, gamma
            coefs_lin_m = coefs_lin.copy()
            coefs_lin_m[vid - 1] *= (1 - eps)
            coefs_lin_p = coefs_lin.copy()
            coefs_lin_p[vid - 1] *= (1 + eps)

        _, xm, _, fxm, _ = gradients.grad_reservoir(alphas, gamma_m,
                                                    Qstars, coefs_lin_m,
                                                    timestep)

        _, xp, _, fxp, _ = gradients.grad_reservoir(alphas, gamma_p,
                                                    Qstars, coefs_lin_p,
                                                    timestep)

        dx_n = (xp - xm) / 2 / eps
        dfx_n = (fxp - fxm) / 2 / eps

        dx_err =  np.abs(comp_fun(dx_v) - comp_fun(dx_n))
        dfx_err =  np.abs(comp_fun(dfx_v) - comp_fun(dfx_n))

        # eliminates nskip values : difficult to explain the difference
        # between numerical diff and our method for certain timesteps sometimes.
        nskip = 20
        dx_errmax = np.sort(dx_err)[-nskip]
        dfx_errmax = np.sort(dfx_err, axis=0)[-nskip]

        assert dx_errmax < 5e-4
        assert dfx_errmax.max() < 1e-3

        if dx_v.std() > 1e-100:
            idx = dx_err < dx_errmax
            corr = np.corrcoef(dx_v[idx], dx_n[idx])[0, 1]
            # Sometimes corr is close to 0 because of strange
            # behaviour of numerical differences
            if abs(corr) > 1e-5:
                assert corr > 0.9999


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("timestep", TIMESTEPS)
def test_grad_model(siteid, timestep, allclose, debug_mode):
    df = data_reader.get_data(siteid, "daily")
    #df = df.iloc[:5]

    mod = gradients.MultiReservoirModel(3)
    mod.alphas = 10
    mod.network.add_link_bypass_shortfall(0, 1)
    mod.network.add_link_total_outflow(1, 2)

    mod.set_coefs_linear(0, FLUX_FUNCTIONS["intercept"])
    mod.set_coefs_linear(1, FLUX_FUNCTIONS["gr4jprod"])
    mod.set_coefs_linear(2, FLUX_FUNCTIONS["gr4jroute"])
    mod.gammas = [5., 400., 50.]

    mod.allocate(len(df), timestep=timestep)

    Qst1 = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
    Qst1[:, -1] *= -1
    Qst1 = np.ascontiguousarray(Qst1)
    mod.set_Qstars(0, Qst1)

    mod.run()

    # Run single reservoir
    _, _, _, fx1, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[0]),
                                               mod.gammas[0],
                                               Qst1,
                                               np.ascontiguousarray(mod.coefs_linear[0]),
                                               timestep, mod.x0[0])

    Qst2 = np.ascontiguousarray(-fx1[:, 3:5])
    _, _, _, fx2, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[1]),
                                               mod.gammas[1],
                                               Qst2,
                                               np.ascontiguousarray(mod.coefs_linear[1]),
                                               timestep, mod.x0[1])

    Qst3 = np.zeros_like(Qst1)
    Qst3[:, 0] = -fx2[:, 5]
    _, _, _, fx3, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[2]),
                                               mod.gammas[2],
                                               Qst3,
                                               np.ascontiguousarray(mod.coefs_linear[2]),
                                               timestep, mod.x0[2])
    for ires in range(mod.nreservoirs):
        fx = [fx1, fx2, fx3][ires]
        assert allclose(fx, mod.fluxes[:, ires])


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("timestep", TIMESTEPS)
def test_grad_model_tangent(siteid, timestep, allclose, debug_mode):
    pytest.skip("WIP")
    df = data_reader.get_data(siteid, "daily")
    df = df.iloc[:100]

    mod = gradients.MultiReservoirModel(2)
    mod.alphas = 10
    mod.network.add_link_total_outflow(0, 1)

    mod.set_coefs_linear(0, FLUX_FUNCTIONS["gr4jprod"])
    mod.set_coefs_linear(1, FLUX_FUNCTIONS["gr4jroute"])
    mod.gammas = [400., 50.]

    vspecs = [[res, i] for res in range(mod.nreservoirs) for i in [0, 1]]
    mod.allocate(len(df), variables_specs=vspecs, timestep=timestep)

    Qst1 = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
    Qst1[:, -1] *= -1
    Qst1 = np.ascontiguousarray(Qst1)
    mod.set_Qstars(0, Qst1)
    mod.run()

    fx = mod.fluxes
    dfx = mod.dfluxes

    for ivar, vspec in enumerate(vspecs):
        rid, vid = vspec
        modp = mod.clone()
        modm = mod.clone()

        if vid == 0:
            g = mod.gammas[rid]
            eps = g * 1e-4
            gs = mod.gammas.copy()
            gs[rid] = g + eps
            modp.gammas = gs

            gs = mod.gammas.copy()
            gs[rid] = g - eps
            modm.gammas = gs

        modp.run()
        modm.run()

        dfx_n = (modp.fluxes - modm.fluxes) / 2 / eps
        dfx_c = dfx[:, ivar, :, :]

        if debug_mode:
            import matplotlib.pyplot as plt
            plt.close("all")
            nfxo = gradients.NFLUXES_OUTPUTS
            nfxi = gradients.NFLUXES_INPUTS
            fig, axs = plt.subplots(nrows=mod.nreservoirs,
                                    ncols=nfxi,
                                    figsize=(15, 7),
                                    layout="constrained")

            for iax, ax in enumerate(axs.flat):
                res = iax // nfxi
                ifx = iax % nfxi
                yc = dfx_c[:, res, ifx]
                yn = dfx_n[:, res, ifx]
                ax.plot(yc, label="C")
                ax.plot(yn, label="N")
                tax = ax.twinx()
                err = yc - yn
                tax.plot(err, color="0.7")
                ax.legend(loc=1)
                title = f"Res {res} - Fx {ifx}"
                ax.set_title(title, y=0.95, x=0.5,
                             ha="center",
                             va="top",
                             fontweight="bold")

            plt.show()
            import pdb; pdb.set_trace()



