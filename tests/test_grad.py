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
from pyquasoare.benchmarks import benchmarks

import c_pyquasoare_grad

import data_reader

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent
LOGGER = iutils.get_logger("models", flog=FTEST / "test_grad.log")
NALPHAS = 50

DEMANDS = [-0.5, 0]
TIMESTEPS = [0.1, 1]
GAMMAS = [10, 100, 10000]
FLUX_SCALINGS = [
    [1.2, 1., 1.],
    [0.8, 1., 1.],
    [1., 1.2, 1.],
    [0.8, 1.2, 1.],
    [1., 1., 1.2]
    ]
VARIABLE_IDS = [0, 1, 2, 3]
NUMDIFF_EPS = 1e-5

def fixed(x, x0):
    return x0 * np.ones_like(x)

def indicator(x, xmin, xmax):
    return np.where((x > xmin) & (x < xmax), 1., 0.)

def comp_fun(x, x0=1e-2):
    return np.asinh(x / x0)

def eff(x, y, nu=1., mini=1e-10):
    tx = comp_fun(x, nu)
    ty = comp_fun(y, nu)
    txm = tx.mean(axis=0)
    denom = ((tx - txm)**2).mean(axis=0)
    out = ((ty - tx)**2).mean() / denom
    if x.ndim == 1:
        out = np.nan if denom < 1e-10 else out
    else:
        out[denom < mini] = np.nan
    return out


FLUX_FUNCTIONS = {
    "gr4jprodquad": [
        lambda x: 1 - np.minimum(x, 1)**2,
        lambda x: np.clip(x, 0, 1) * (2 - np.clip(x, 0, 1)),
        lambda x: 4**3/9**4 * np.maximum(x, 0)**2
        ],
    "gr4jroutequad": [
        lambda x: 0.9 * fixed(x, 1.),
        lambda x: np.maximum(x, 0.)**2,
        lambda x: np.maximum(x, 0.)**2 / 4
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
    "muskingumrouting": [
        # KM = 2.5 / XM = 0.2
        lambda x: indicator(x, 0, np.inf) / (1 - 0.2),
        lambda x: fixed(x, 1.),
        lambda x: np.maximum(x, 0.) / 2.5 / (1 - 0.2)
        ],
}

FLUX_FUNCTIONS.update(gradients.FLUX_FUNCTIONS)


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

    rid = 0
    vrid = np.array([0] * nvars, dtype=np.int32)
    vtid = np.array([0] * nvars, dtype=np.int32)
    qst = -np.ones(2)
    dqst = np.zeros((nvars, gradients.NFLUXES_INPUTS))
    fxs = np.ones(3)
    t0 = 0.
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, alphas, timestep, gamma,
                                            ipm, ipmw, qst, dqst, fxs, coefs_lin,
                                            vrid, vtid, niter, x, dx, fx, dfx)
    assert ierr == 112

    qst = np.ones(2)
    coefs_lin[0] *= -1
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, alphas, timestep, gamma,
                                            ipm, ipmw, qst, dqst, fxs, coefs_lin,
                                            vrid, vtid, niter, x, dx, fx, dfx)
    assert ierr == 111

    coefs_lin[0] *= -1
    gamma = 0
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, alphas, timestep, gamma,
                                            ipm, ipmw, qst, dqst, fxs, coefs_lin,
                                            vrid, vtid, niter, x, dx, fx, dfx)
    assert ierr == 113

    gamma = 100.
    vtid = np.array([1] * nvars, dtype=np.int32)
    fxs = np.zeros(3)
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, alphas, timestep, gamma,
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
    start, end = 0, 500
    Qstars, scalings = get_data(siteid, reservoir, timestep,
                                gamma, demand, start, end)
    alphas, coefs, coefs_lin = get_alphas_and_coefs(reservoir, scalings)

    nres = 3
    ipm, ipmw = get_input_mapping(nres, "bypass_shortfall")

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
                                                ipm, ipmw, qst, dqst, fxs, coefs_lin,
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
@pytest.mark.parametrize("flux_scalings", FLUX_SCALINGS)
def test_grad_reservoir(siteid, reservoir, gamma, timestep, demand,
                        flux_scalings, allclose):
    skip_this(reservoir, gamma, demand)

    Qstars, scalings = get_data(siteid, reservoir, timestep, gamma, demand)
    alphas, coefs, coefs_lin = get_alphas_and_coefs(reservoir, scalings)

    # Flux scalings
    fxs = np.array(flux_scalings)

    # Quadratic
    x0 = 0.5
    coefs[0] *= fxs[0]
    coefs[1] *= fxs[1]
    coefs[2] *= fxs[2]
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
    assert allclose(fx2[:, 3], inflow - Qstars[:, 0])
    assert allclose(fx2[:, 4], demand - Qstars[:, 1])
    assert allclose(fx2[:, 5], fx2[:, 2] + fx2[:, 3])

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
@pytest.mark.parametrize("flux_scalings", FLUX_SCALINGS)
def test_grad_reservoir_tangent(siteid, reservoir, gamma, timestep,
                                demand, flux_scalings,
                                allclose, debug_mode):
    skip_this(reservoir, gamma, demand)
    start, end = None, None
    Qstars, scalings = get_data(siteid, reservoir, timestep, gamma, demand,
                                start, end)
    alphas, coefs, coefs_lin = get_alphas_and_coefs(reservoir, scalings)

    # Flux scalings
    fxs = np.array(flux_scalings)

    # Gradient
    n, x, dx, fx, dfx = gradients.grad_reservoir(alphas, gamma,
                                                 Qstars, coefs_lin,
                                                 timestep,
                                                 flux_scalings=fxs,
                                                 errors="raise")

    # Numerical difference for each variable
    for vid in range(gradients.VARTYPE_NUMBER):
        dx_v = dx[:, vid]
        dfx_v = dfx[:, vid]

        if vid == 0:
            eps = gamma * 1e-8
            gamma_m = gamma - eps
            gamma_p = gamma + eps
            coefs_lin_m = coefs_lin
            coefs_lin_p = coefs_lin
        else:
            eps = 1e-6
            gamma_m, gamma_p = gamma, gamma
            coefs_lin_m = coefs_lin.copy()
            coefs_lin_m[vid - 1] *= (1 - eps)
            coefs_lin_p = coefs_lin.copy()
            coefs_lin_p[vid - 1] *= (1 + eps)

        _, xm, _, fxm, _ = gradients.grad_reservoir(alphas, gamma_m,
                                                    Qstars, coefs_lin_m,
                                                    timestep,
                                                    flux_scalings=fxs)

        _, xp, _, fxp, _ = gradients.grad_reservoir(alphas, gamma_p,
                                                    Qstars, coefs_lin_p,
                                                    timestep,
                                                    flux_scalings=fxs)

        dx_n = (xp - xm) / 2 / eps
        dfx_n = (fxp - fxm) / 2 / eps

        if vid == 0:
            # Convert dflux against gamma to dflux against gamma scaling
            dfx_n *= gamma
            dfx_v *= gamma

        dx_eff = eff(dx_n, dx_v)
        dx_emax = np.abs(dx_n - dx_v).max()

        dfx_eff = eff(dfx_n, dfx_v)
        dfx_emax = np.abs(dfx_n - dfx_v).max()

        # Strong condition > if one of the variable has nan eff, then skip
        # this excludes test for the others
        if np.any(np.isnan(dfx_eff)) or np.isnan(dx_eff):
            continue

        xtol = 1e-8
        ftol = 5e-6
        if debug_mode and (dx_eff > xtol or np.nanmin(dfx_eff) > ftol):
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(ncols=3, nrows=2,
                                    figsize=(12, 7),
                                    layout="constrained")
            for ifx, ax in enumerate(axs.flat):
                v = comp_fun(dfx_v[:, ifx])
                n = comp_fun(dfx_n[:, ifx])
                ax.plot(v)
                ax.plot(n)
                tax = ax.twinx()
                tax.plot(np.abs(v - n), color="0.5", lw=0.8, alpha=0.5)
                #tax.plot(fx[:, ifx], color="0.5", lw=0.8, alpha=0.5)

                e = eff(n, v)
                m = np.abs(n - v).max()
                title = f"Flux {ifx} - eff={e:0.1e} emax={m:0.1e}"
                ax.set_title(title)

            ftitle = f"Diff against dvar={vid} eps={eps:0.1e}"
            fig.suptitle(ftitle, fontweight="bold")
            plt.show()
            import pdb; pdb.set_trace()

        assert dx_eff < xtol
        assert np.nanmax(dfx_eff) < ftol


def test_grad_model_setters(allclose):
    siteid = "203014"
    timestep = 1.
    mod = gradients.MultiReservoirModel(2, errors="raise")
    mod.alphas = 10
    mod.network.add_link_bypass_shortfall(0, 1)
    mod.set_coefs_linear(0, FLUX_FUNCTIONS["openwater"])
    mod.set_coefs_linear(1, "gr4jprod")

    mod.gammas = 10
    assert allclose(mod.gammas, 10.)

    mod.gammas = [100., 10]
    assert allclose(mod.gammas, [100., 10])

    mod.flux_scalings = 1.2
    assert allclose(mod.flux_scalings, 1.2)

    fxs = np.ones((mod.nreservoirs, 3))
    fxs[0, 0] = 1.2
    mod.flux_scalings = fxs
    assert allclose(mod.flux_scalings, fxs)


def test_grad_model_clone(allclose):
    mod = gradients.MultiReservoirModel(2, errors="raise")
    mod.alphas = 10
    mod.network.add_link_bypass_shortfall(0, 1)
    mod.set_coefs_linear(0, "openwater")
    mod.set_coefs_linear(1, "gr4jprod")
    mod.gammas = [5., 400.]

    siteid = "203014"
    timestep = 1.
    df = data_reader.get_data(siteid, "daily")
    mod.allocate(len(df), timestep=timestep)
    Qst1 = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
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
@pytest.mark.parametrize("flux_scalings", [
        [1., 1., 1.],
        [1., 1.2, 1.]
    ])
def test_grad_model(siteid, timestep, allclose, flux_scalings, debug_mode):
    mod = gradients.MultiReservoirModel(6, errors="ignore")

    mod.alphas = 10
    mod.network.add_link_bypass_shortfall(0, 1)
    mod.network.add_link_outflow(1, 2)
    mod.network.add_link_outflow(2, 5)
    mod.network.add_link_outflow(3, 5)
    mod.network.add_link_outflow(4, 5)

    mod.set_coefs_linear(0, "openwater")
    mod.set_coefs_linear(1, "gr4jprod")
    mod.set_coefs_linear(2, "gr4jroute")
    mod.set_coefs_linear(3, "gr4jroute")
    mod.set_coefs_linear(4, "gr4jroute")
    mod.set_coefs_linear(5, "transluent")
    mod.gammas = [5., 400., 50., 10., 5., 1.]

    fxs = np.ones((mod.nreservoirs, 3))
    fxs[0] = flux_scalings
    mod.flux_scalings = fxs

    df = data_reader.get_data(siteid, "daily")
    df = df.iloc[:1]

    nrep = 5
    t = timestep * (np.arange(len(df))[:, None] + np.linspace(1./nrep, 1, nrep)[None, :]).ravel()
    df = pd.DataFrame(np.repeat(df, nrep, axis=0),
                      columns=df.columns)
    timestep /= nrep

    mod.allocate(len(df), timestep=timestep)
    Qst0 = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values.copy()
    Qst0[:, -1] *= -1
    Qst0 = np.ascontiguousarray(Qst0)
    mod.set_Qstars(0, Qst0)

    Qst3 = Qst0.copy()
    Qst3[:, 1] = 0.
    mod.set_Qstars(3, Qst3)
    mod.set_Qstars(4, Qst3)

    mod.run()

    # Run single reservoirs
    coefs = np.ascontiguousarray(mod.coefs_linear[0])
    coefs[0] *= flux_scalings[0]
    coefs[1] *= flux_scalings[1]
    coefs[2] *= flux_scalings[2]
    _, x0, _, fx0, _ = gradients.grad_reservoir(np.ascontiguousarray(mod.alphas[0]),
                                               mod.gammas[0],
                                               Qst0,
                                               coefs,
                                               timestep, mod.x0[0])
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
@pytest.mark.parametrize("demand", DEMANDS)
def test_grad_model_tangent(siteid, timestep, demand, allclose, debug_mode):
    df = data_reader.get_data(siteid, "daily")
    df = df.iloc[:20]

    mod = gradients.MultiReservoirModel(3, "raise")
    mod.alphas = 10
    mod.network.add_link_bypass_shortfall(0, 1)
    mod.network.add_link_outflow(1, 2)

    mod.set_coefs_linear(0, "gr4jprod")
    mod.set_coefs_linear(1, "gr4jprod")
    mod.set_coefs_linear(2, "gr4jroute")
    mod.gammas = [50., 400., 50.]

    nv = gradients.VARTYPE_NUMBER
    vspecs = [[res, i] for res in range(mod.nreservoirs) for i in range(nv)]
    mod.allocate(len(df), variables_specs=vspecs, timestep=timestep)

    Qst1 = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
    Qst1[:, -1] *= -1
    Qst1 = np.ascontiguousarray(Qst1)
    mod.set_Qstars(0, Qst1)

    Qst2 = np.zeros((len(df), 2))
    Qst2[:, -1] = 1.
    mod.set_Qstars(2, Qst2)

    # Set demand
    fxs = np.ones((mod.nreservoirs, 3))
    fxs[-1, 1] = max(demand, 1e-10)
    mod.flux_scalings = fxs

    mod.run()
    fx = mod.fluxes
    dfx = mod.dfluxes

    # Numerical differences
    for ivar, vspec in enumerate(vspecs):
        rid, vid = vspec

        # Test downstream parameters have no effect on upstream fluxes
        if rid > 0:
            dfxu = dfx[:, ivar, :rid, :]
            assert allclose(dfxu, 0.)

        modp = mod.clone()
        modp.allocate(len(df), variables_specs=None, timestep=timestep)
        modm = mod.clone()
        modm.allocate(len(df), variables_specs=None, timestep=timestep)

        if vid == 0:
            # Differentiate against gamma
            gs = mod.gammas.copy()
            eps = gs[rid] * 1e-4
            gs[rid] = gs[rid] + eps
            modp.gammas = gs

            gs = mod.gammas.copy()
            gs[rid] = gs[rid] - eps
            modm.gammas = gs
        else:
            eps = NUMDIFF_EPS
            coefs_lin = modm._coefs_linear
            coefs_lin[rid, vid - 1] *= (1 - eps)

            d = modm.coefs_linear[rid, vid - 1] - mod.coefs_linear[rid, vid - 1]

            coefs_lin = modp._coefs_linear
            coefs_lin[rid, vid - 1] *= (1 + eps)

        modp.run()
        modm.run()

        dfx_n = (modp.fluxes - modm.fluxes) / 2 / eps
        dfx_c = dfx[:, ivar, :, :]
        atol = 1e-7

        assert allclose(dfx_n, dfx_c, atol=atol)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("gamma", [10, 100, 10000])
@pytest.mark.parametrize("timestep", [1., 0.1])
def test_grad_model_vs_gradbench_quadrouting(siteid, gamma, timestep,
                                             debug_mode, allclose):
    noriginal = 365 * 5
    nrepeat = 500 #500
    df = data_reader.get_data(siteid, "daily")[-noriginal:]
    df = df.interpolate().iloc[-noriginal:]
    assert df.shape[0] == noriginal
    df = np.tile(df, (nrepeat, 1))
    pin = df[:, 0]
    ein = df[:, 1]
    qin = df[:, 2]
    nval = len(df)

    x0 = 0.5

    # run benchmarks
    gammas = np.array([gamma], dtype=np.float64)
    dgammas = np.array([1.])
    t0 = time.time()
    out = benchmarks.gradbench_quadrouting_tangent(timestep, qin, x0,
                                                   gammas, dgammas)
    dtb = time.time() - t0

    x, dx, fx, dfx = out

    # run quasoare
    errors = "raise"
    mod = gradients.MultiReservoirModel(1, errors=errors)

    # super careful with alphas around 0!!!!!
    nalphas = 500
    nu = 1e-4
    alphas = np.array([-nu, 0., nu] + np.linspace(2 * nu, x.max(), nalphas - 3).tolist())
    mod.alphas = alphas[None, :]
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

    # numerical diff on quasoare
    modm = mod.clone()
    modm.allocate(nval, timestep=timestep)
    modp = mod.clone()
    modp.allocate(nval, timestep=timestep)

    eps = 1e-3 # becomes worse if eps < 1e-12, optimal for 1e-2
    modm.gammas = gamma - eps
    modp.gammas = gamma + eps

    t0 = time.time()
    modm.run()
    modp.run()
    dtn = time.time() - t0

    dx_n = (modp.x - modm.x) / eps / 2
    dfx_n = (modp.fluxes - modm.fluxes) / eps / 2

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

        import matplotlib.pyplot as plt
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
            ax.plot(wb, "k-", label="benchmark")
            ax.plot(wq, "k--", label="quasoare")

            atol = 1e-5
            eff_q = eff(wb, wq, atol)
            ax.set(title=f"{title} perf: q={eff_q:2.1e}")
            ax.legend(loc=1)

            ax = axs[1, ivar]
            ax.plot(wq - wb, "k--", label="quasoare", lw=0.9, color="0.5")
            ax.legend(loc=1)

            ax = axs[2, ivar]
            ax.plot(dwb, "k-", label="benchmark")
            ax.plot(dwq, "k--", label="quasoare")
            ax.plot(dwn, "-", label="numerical", color="tab:blue")

            eff_q = eff(dwb, dwq, atol)
            eff_n = eff(dwb, dwn, atol)
            title = f"{title} gradient perf: q={eff_q:2.1e} n={eff_n:2.1e}"
            ax.set(title=title)
            ax.legend(loc=1)

            ax = axs[3, ivar]
            ax.plot(dwq - dwb, "k--", label="quasoare", lw=0.9, color="0.5")
            ax.plot(dwn - dwb, "-", label="numerical", lw=0.9, color="tab:blue")
            ax.legend(loc=1)

        plt.show()
        import pdb; pdb.set_trace()

    atol = 1e-5
    eff_fx = eff(fx[:, 0], -mod.fluxes[:, 0, 5], atol)
    eff_dfx = eff(dfx[:, 0], -mod.dfluxes[:, 0, 0, 5], atol)
    eff_dfxn = eff(dfx[:, 0], -dfx_n[:, 0, 5], atol)

    print(f"\nTime : dtq={dtq:2.1e} dtn={dtn:2.1e} / ratio={dtq/dtn:0.1e}")
    print(f"Flux  perf: q={eff_fx:2.1e}")
    print(f"dFlux perf: q={eff_dfx:2.1e} n={eff_dfxn:2.1e} / ratio={eff_dfx/eff_dfxn:0.1e}")

    assert eff_fx < 2e-4
    assert eff_dfx < 5e-3
    assert eff_dfx < eff_dfxn * 1.05

    assert dtq < dtb * 30
    assert dtq < dtn * 0.66


@pytest.mark.parametrize("gamma", [10, 1000])
@pytest.mark.parametrize("demand", [-0.1])
@pytest.mark.parametrize("timestep", TIMESTEPS)
@pytest.mark.parametrize("nreaches", [1, 2, 5])
@pytest.mark.parametrize("diff", [1, 2, 3]) # 4 does not work yet
def test_grad_model_vs_gradbench_semidist(gamma, demand, timestep, nreaches,
                                          diff, allclose, debug_mode):
    #pytest.skip("WIP")
    noriginal = 365 * 5
    nrepeat = 100
    nval = nrepeat * noriginal
    pin = np.empty((nval, nreaches))
    ein = np.empty((nval, nreaches))
    # Random catchment areas
    areas = np.linspace(100, 500, nreaches)
    for isite, siteid in enumerate(data_reader.SITEIDS[:nreaches]):
        df = data_reader.get_data(siteid, "daily")[-noriginal:]
        assert df.shape[0] == noriginal
        df = df.interpolate().iloc[-noriginal:]
        df = np.tile(df, (nrepeat, 1))
        pin[:, isite] = df[:, 0]
        ein[:, isite] = df[:, 1]

    # run benchmarks
    x0 = 0.5
    gammas_rr_prod = np.array([gamma] * nreaches)
    zr = np.zeros(nreaches)
    o = np.array([1.] + [0.] * (nreaches - 1))
    dgammas_rr_prod = o if diff == 1 else zr
    gammas_rr_route = np.array([gamma / 10] * nreaches)
    dgammas_rr_route = o if diff == 2 else zr
    gammas_ro_route = np.array([gamma / 20] * nreaches)
    dgammas_ro_route = o if diff == 3 else zr
    demands_rr_route = np.ones(nreaches) * demand
    ddemands_rr_route = o if diff == 4 else zr

    t0 = time.time()
    out = benchmarks.gradbench_semidist_tangent(timestep, pin, ein, x0, areas,
                                                gammas_rr_prod,
                                                dgammas_rr_prod,
                                                gammas_rr_route,
                                                dgammas_rr_route,
                                                gammas_ro_route,
                                                dgammas_ro_route,
                                                demands_rr_route,
                                                ddemands_rr_route)
    dtb = time.time() - t0
    x_rr_prod, dx_rr_prod, \
        x_rr_route, dx_rr_route, \
        x_ro_route, dx_ro_route, \
        fx_rr_prod, dfx_rr_prod, \
        fx_rr_route, dfx_rr_route, \
        fx_ro_route, dfx_ro_route = out

    # run quasoare
    nres_per_reach = 4
    nres = nreaches * nres_per_reach
    mod = gradients.MultiReservoirModel(nres, errors="raise")


    # .. gradients against capacity in top reach
    #    if diff <= 3 or demand if diff == 4 (i.e. demand flux scaling)
    vspecs = [[diff - 1, 0]] if diff != 4 else [[1, 2]]

    # (test gradients against all variables)
    #vspecs = [[r, t] for r in range(mod.nreservoirs) for t in range(4)]
    #print(f"\n{len(vspecs)} variables")

    mod.allocate(nval, timestep=timestep,
                 variables_specs=vspecs)

    # Set alphas
    nalphas = 500
    xmaxs = np.zeros(mod.nreservoirs)
    for ireach in range(nreaches):
        xmaxs[nres_per_reach * ireach] = np.max(x_rr_prod[:, ireach])
        xmaxs[nres_per_reach * ireach + 1] = np.max(x_rr_route[:, ireach])
        xmaxs[nres_per_reach * ireach + 2] = np.max(x_ro_route[:,  ireach])
        xmaxs[nres_per_reach * ireach + 3] = 1

    nu = 1e-5
    lows = np.array([-nu, 0., nu / 10, nu])

    alphas = [np.repeat(lows[None, :], mod.nreservoirs, 0),
              2 * nu + xmaxs[:, None] * np.linspace(0, 1.,
                                                    nalphas - 4)[None, :]]
    mod.alphas = np.column_stack(alphas)

    # Set data and network
    for ireach in range(nreaches):
        ibase = nres_per_reach * ireach
        # RR prod = ibase
        # RR routing = ibase + 1
        # RO routing = ibase + 2
        # Transluent = ibase + 3

        # .. inputs ..
        # Set RR prod inputs
        Qst = np.column_stack([pin[:, ireach], -ein[:, ireach]])
        mod.set_Qstars(ibase, Qst)

        # Set RR route inputs
        Qst = np.ones((nval, 2))
        Qst[:, 0] = np.nan
        mod.set_Qstars(ibase + 1, Qst)

        # .. coefs ...
        # Set RR prod coefs
        mod.set_coefs_linear(ibase, FLUX_FUNCTIONS["gr4jprodquad"])

        # Set RR route coefs
        mod.set_coefs_linear(ibase + 1, FLUX_FUNCTIONS["gr4jroutequad"])

        # Set RO route coefs
        mod.set_coefs_linear(ibase + 2, FLUX_FUNCTIONS["quadrouting"])

        # Set Transluent coefs
        mod.set_coefs_linear(ibase + 3, "transluent")

        # .. links ...
        # RR prod -> RR routing
        mod.network.add_link_outflow(ibase, ibase + 1)

        # RR route -> transluent with conversion
        conv = areas[ireach] * timestep / 86.4
        mod.network.add_link_outflow(ibase + 1, ibase + 3, conv)

        # Transluent upstream -> RO route
        if ireach > 0:
            mod.network.add_link_outflow(ibase - 1, ibase + 2)

        # RO Route -> transluent
        mod.network.add_link_outflow(ibase + 2, ibase + 3)

    # Set capacity
    mod.gammas = np.tile([gamma, gamma / 10, gamma / 20, 1.], (nreaches,))

    # Set demands for rr route reservoirs
    mod.flux_scalings[1::nres_per_reach, 1] = demand

    # Run model
    t0 = time.time()
    mod.run()
    dtq = time.time() - t0

    if debug_mode:
        import matplotlib.pyplot as plt
        plt.close("all")
        # Skip model routing reservoir (replaced by transluent)
        nres = (nres_per_reach - 1) * nreaches
        aw, ah = 4, 3
        fig, axs = plt.subplots(nrows=2, ncols=nres,
                                figsize=(nres * aw, 2 * ah), layout="constrained",
                                sharex=True)

        for res in range(nres):
            ireach = res // 3
            ires_mod_base = nres_per_reach * ireach

            if res % 3 == 0:
                yb = fx_rr_prod[-noriginal:, ireach]
                dyb = dfx_rr_prod[-noriginal:, ireach]
                rtype = "RR prod"
                ires_mod = ires_mod_base
            elif res % 3 == 1:
                yb = fx_rr_route[-noriginal:, ireach]
                dyb = dfx_rr_route[-noriginal:, ireach]
                rtype = "RR route"
                ires_mod = ires_mod_base + 1
            elif res % 3 == 2:
                yb = fx_ro_route[-noriginal:, ireach]
                dyb = dfx_ro_route[-noriginal:, ireach]
                rtype = "RO route"
                ires_mod = ires_mod_base + 3

            yq = -mod.fluxes[-noriginal:, ires_mod, 5]
            dyq = -mod.dfluxes[-noriginal:, 0, ires_mod, 5]

            if diff < 4:
                g0 = mod.gammas[diff]
                dyb *= g0
                dyq *= g0

            ax = axs[0, res]
            ax.plot(yb, label="benchmark")
            ax.plot(yq, label="quasoare")
            tax = ax.twinx()

            err = np.abs(yb - yq)
            tax.plot(err, lw=0.8, color="0.6", alpha=0.2)
            ax.text(0.02, 0.98, f"Emax={err.max():2.2e}",
                    va="top", ha="left", transform=ax.transAxes)

            title = f"R{res} ({rtype})"
            ylabel = "Flow [m3/s]" if res % 3 == 2 else "Runoff [mm/d]"
            ax.set(title=title, ylabel=ylabel)
            if res == 0:
                ax.legend()

            ax = axs[1, res]
            ax.plot(dyb)
            ax.plot(dyq)
            tax = ax.twinx()
            err = np.abs(dyb - dyq)
            tax.plot(err, lw=0.8, color="0.6", alpha=0.2)
            ax.text(0.02, 0.98, f"Emax={err.max():2.2e}",
                    va="top", ha="left", transform=ax.transAxes)

            ax.set(ylabel=ylabel)

        ftitle = f"Diff {diff} Δb={dtb:4.2e}s Δq={dtq:4.2e}s"
        fig.suptitle(ftitle, fontweight="bold", fontsize=15)
        fp = FTEST / "sim.png"
        fig.savefig(fp)

    print(f"\nTime: dtb={dtb:4.2e} dtq={dtq:4.2e} / ratio = {dtq / dtb:0.1f}")
    assert dtq < dtb * 65

    # Error on rr prod
    nrp = nres_per_reach
    err = np.abs(comp_fun(fx_rr_prod) + comp_fun(mod.fluxes[:, ::nrp, 5]))
    logtxt = f"FRP={err.max():3.1e}"
    assert err.max() < 5e-5

    err = np.abs(comp_fun(dfx_rr_prod) + comp_fun(mod.dfluxes[:, 0, ::nrp, 5]))
    logtxt += f" DRP={err.max():3.1e}"
    assert err.max() < 5e-4

    # Error on rr route
    err = np.abs(comp_fun(fx_rr_route) + comp_fun(mod.fluxes[:, 1::nrp, 5]))
    logtxt += f" FRR={err.max():3.1e}"
    assert err.max() < 5e-3

    err = np.abs(comp_fun(dfx_rr_route) + comp_fun(mod.dfluxes[:, 0, 1::nrp, 5]))
    logtxt += f" DRR={err.max():3.1e}"
    assert err.max() < 2e-2 # this is quite high

    # Error on ro route
    err = np.abs(comp_fun(fx_ro_route) + comp_fun(mod.fluxes[:, 3::nrp, 5]))
    logtxt += f" FRO={err.max():3.1e}"
    assert err.max() < 5e-3

    err = np.abs(comp_fun(dfx_ro_route) + comp_fun(mod.dfluxes[:, 0, 3::nrp, 5]))
    logtxt += f" DRO={err.max():3.1e}"
    assert err.max() < 1e-2

    print(f"{logtxt}\n")
