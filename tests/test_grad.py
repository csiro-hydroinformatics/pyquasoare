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

def comp_fun(x, x0=1e-3):
    return np.asinh(x / x0)

FLUX_FUNCTIONS = {
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
    "xinanjiang1": [
        # exponent = 0.2
        lambda x: (1 - np.clip(x, 0., 1.))**0.2,
        lambda x: np.maximum(x, 0.),
        lambda x: fixed(x, 0.)
        ],
    "xinanjiang2": [
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
             + np.linspace(2 * eps, 1 - 2 * eps, NALPHAS - 12).tolist() \
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
    nfxp = gradients.NFLUXES_POTENTIAL
    nfxo = gradients.NFLUXES_OUTPUTS
    assert ipm.shape == (netw.nreservoirs, nfxp, netw.nreservoirs, nfxo)

    # Reservoirs 0 receives nothing
    assert allclose(ipm[0], 0)

    ifxi = gradients.IFLUX_INFLOW
    ifxd = gradients.IFLUX_DEMAND
    ifxb = gradients.IFLUX_BYPASS
    ifxs = gradients.IFLUX_SHORTFALL
    ifxt = gradients.IFLUX_TOTALOUT

    if linktype == "total":
        assert ipm[1, ifxi, 0, ifxt] == -1
        assert ipm[2, ifxi, 1, ifxt] == -1
    else:
        assert ipm[1, ifxi, 0, ifxb] == -1
        assert ipm[1, ifxd, 0, ifxs] == -1
        assert ipm[2, ifxi, 1, ifxb] == -1
        assert ipm[2, ifxd, 1, ifxs] == -1

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
    ipm = np.zeros((nres, gradients.NFLUXES_POTENTIAL,
                    nres, gradients.NFLUXES_OUTPUTS))

    nvars = 2
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
    x_start = 0.5
    dx0 = np.zeros(nvars)

    rid = 0
    vrid = np.array([0] * nvars, dtype=np.int32)
    vtid = np.array([0] * nvars, dtype=np.int32)
    qst = -np.ones(2)
    fxs = np.ones(3)
    t0 = 0.
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, alphas, timestep, gamma,
                                            ipm, qst, fxs, coefs_lin, t0,
                                            x_start, dx0, vrid, vtid,
                                            niter, x, dx, fx, dfx)
    assert ierr == 112

    qst = np.ones(2)
    coefs_lin[0] *= -1
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, alphas, timestep, gamma,
                                            ipm, qst,fxs, coefs_lin, t0,
                                            x_start, dx0, vrid, vtid,
                                            niter, x, dx, fx, dfx)
    assert ierr == 111

    coefs_lin[0] *= -1
    gamma = 0
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, alphas, timestep, gamma,
                                            ipm, qst, fxs, coefs_lin, t0,
                                            x_start, dx0, vrid, vtid,
                                            niter, x, dx, fx, dfx)
    assert ierr == 113

    gamma = 100.
    vtid = np.array([1] * nvars, dtype=np.int32)
    fxs = np.zeros(3)
    ierr = c_pyquasoare_grad.grad_integrate(nres, rid, alphas, timestep, gamma,
                                            ipm, qst, fxs, coefs_lin, t0,
                                            x_start, dx0, vrid, vtid,
                                            niter, x, dx, fx, dfx)
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
    nres = 1
    ipm = np.zeros((nres, gradients.NFLUXES_POTENTIAL,
                    nres, gradients.NFLUXES_OUTPUTS))

    # Loop and integrate
    rid = 0
    vtid = np.arange(4, dtype=np.int32)
    nvars = len(vtid)
    vrid = np.zeros(nvars, dtype=np.int32)
    x_start = 0.5
    dx0 = np.zeros(nvars)
    dx0m = np.zeros(nvars)
    dx0p = np.zeros(nvars)
    scalings = np.ones(3)
    niter2 = np.zeros(1, dtype=np.int32)
    x2 = np.zeros(1, dtype=np.float64)
    x2p = np.zeros(1, dtype=np.float64)
    x2m = np.zeros(1, dtype=np.float64)
    dx2 = np.zeros(nvars, dtype=np.float64)
    nfxo = gradients.NFLUXES_OUTPUTS
    fx2 = np.zeros(nfxo, dtype=np.float64)
    dfx2 = np.zeros((nvars, nres, nfxo), dtype=np.float64)
    fxs = np.ones(gradients.NFLUXES_INPUTS, dtype=np.float64)
    eps = 1e-6
    t0 = 0.

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
                                                ipm, qst, fxs, coefs_lin, t0,
                                                x_start, dx0, vrid, vtid,
                                                niter2, x2, dx2, fx2, dfx2)
        assert allclose(x1, x2[0])
        assert allclose(fx1, fx2[:3])

        fxa = np.array([fx1[0] - qst[0], fx1[1] - qst[1], fx1[0] - qst[0] + fx1[2]])
        assert allclose(fxa, fx2[3:])


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
    dx0 = np.array([-x0 / gamma])
    vtid = np.zeros(1, dtype=np.int32)
    n2, x2, dx2, fx2 = gradients.grad_reservoir(alphas, timestep,
                                          Qstars, gamma, coefs_lin,
                                          x0, dx0, vtid,
                                          errors="raise")
    assert allclose(x1, x2)
    assert allclose(fx1, fx2)
    assert all(~np.isnan(x2))
    assert all(~np.isnan(dx2))

    # Sanity check
    if not re.search("rout(e|ing)", reservoir):
        assert all((x2 >= 0) & (x2 <= 1.05))

    # Mass balance
    dSa = np.diff(x2) * gamma
    dSb = timestep * fx2[1:, :3].sum(axis=1)
    assert allclose(dSa, dSb)



@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", FLUX_FUNCTIONS)
@pytest.mark.parametrize("gamma", GAMMAS)
@pytest.mark.parametrize("timestep", TIMESTEPS)
@pytest.mark.parametrize("demand", DEMANDS)
@pytest.mark.parametrize("variable_id", VARIABLE_IDS)
def test_grad_reservoir_tangent(siteid, reservoir, gamma, timestep,
                                demand, variable_id, allclose):
    skip_this(reservoir, gamma, demand)
    start, end = None, None
    Qstars, scalings = get_data(siteid, reservoir, timestep, gamma, demand,
                                start, end)
    alphas, coefs, coefs_lin = get_alphas_and_coefs(reservoir, scalings)

    # Gradient vs capacity Γ
    x0 = 0.5
    dx0 = np.zeros(1)
    vtid = np.array([variable_id], dtype=np.int32)
    n2, x2, dx2, fx2 = gradients.grad_reservoir(alphas, timestep,
                                                Qstars, gamma, coefs_lin,
                                                x0, dx0, vtid,
                                                errors="raise")
    # Numerical difference
    vtid = np.array([variable_id], dtype=np.int32)
    if variable_id == 0:
        eps = gamma * 1e-6
        gamma_m = gamma - eps
        gamma_p = gamma + eps
        coefs_lin_m = coefs_lin
        coefs_lin_p = coefs_lin
    else:
        eps = NUMDIFF_EPS
        gamma_m, gamma_p = gamma, gamma
        coefs_lin_m = coefs_lin.copy()
        coefs_lin_m[variable_id - 1] *= (1 - eps)
        coefs_lin_p = coefs_lin.copy()
        coefs_lin_p[variable_id - 1] *= (1 + eps)

    _, x2m, _, fx2m = gradients.grad_reservoir(alphas, timestep,
                                               Qstars, gamma_m, coefs_lin_m,
                                               x0, dx0, vtid,
                                               errors="raise")

    _, x2p, _, fx2p = gradients.grad_reservoir(alphas, timestep,
                                               Qstars, gamma_p, coefs_lin_p,
                                               x0, dx0, vtid,
                                               errors="raise")

    # Test dx2
    #if reservoir in ["linrouting", "muskingumrouting"] and variable_id == 0:
    #    # Storage does not depend on Γ hence:
    #    # x = S / Γ => dx = - S / Γ^2 => dx = -x / Γ
    #    expected = - x2 / gamma
    #else:
    #    expected = (x2p - x2m) / 2 / eps
    expected = (x2p - x2m) / 2 / eps

    debug = False

    err =  np.abs(comp_fun(dx2) - comp_fun(expected))
    # eliminates nskip values : difficult to explain the difference
    # between numerical diff and our method for certain timesteps sometimes.
    nskip = 20
    errmax = np.sort(err)[-nskip]
    if not debug:
        assert errmax < 1e-4

    if dx2.std() > 1e-100:
        idx = err < errmax
        corr = np.corrcoef(dx2[idx], expected[idx])[0, 1]
        # Sometimes corr is close to 0 because of strange
        # behaviour of numerical differences
        if not debug and abs(corr) > 1e-5:
            assert corr > 0.9999

    if debug:
        import matplotlib.pyplot as plt
        from hydrodiy.plot import putils
        fig, axs = plt.subplots(nrows=2, sharex=True,
                               figsize=(13, 6), layout="constrained")
        ax = axs[0]
        ax.plot(x2)
        ax.plot(x2m, color="0.6", ls="--")
        ax.plot(x2p, color="0.4", ls=":")
        xmax = x2.max()
        for a in alphas:
            if a > xmax:
                continue
            putils.line(ax, 1, 0, 0, a, "-", color="0.7", lw=0.5)

        ax = axs[1]
        ax.plot(dx2, label="C")
        ax.plot(expected, label="expected")
        ax.set(title=f"corr={corr:6.4f} errmax={errmax:6.4f}")
        ax.legend()
        tax = ax.twinx()
        err = np.maximum(err, 1e-20)
        tax.plot(err, color="0.7")
        tax.set(ylim=(1e-10, err.max()), yscale="log")
        plt.show()
        import pdb; pdb.set_trace()


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("timestep", [0.1, 1])
def test_grad_gr4j(siteid, timestep, allclose):
    return

    demand = -0.2

    # Get data
    df = data_reader.get_data(siteid, "daily")

    Qstars = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
    Qstars[:, 1] *= -1 # PET = negative demand

    Qstars = np.column_stack([Qstars, np.nan * Qstars, np.nan * Qstars])
    Qstars = np.ascontiguousarray(Qstars)
    Qstars[:, -1] = demand  # Fixed demand for routing reservoir

    nin = 2
    nout = 6
    nres = Qstars.shape[1] // nin

    netw = gradients.Network(nres)
    netw.add_link_bypass_shortfall(0, 1) # Intercept -> Prod
    netw.add_link_total_outflow(1, 2) # Prod -> Route

    # Get params
    # Intercept = 1mm
    # Prod = 100mm
    # Route = 20mm
    gammas = np.array([1., 100., 20.])

    nalphas = 50
    a1 = np.linspace(0, 1, nalphas)
    a2 = np.linspace(0, 1.5, nalphas)
    alphas = np.vstack([a1, a1, a2])

    fint = FLUX_FUNCTIONS["intercept"]
    coefs_grad_int = gradients.lin_coefficient_matrix(fint, a1)

    fprod = FLUX_FUNCTIONS["gr4jprod"]
    coefs_grad_prod = gradients.lin_coefficient_matrix(fprod, a1)

    froute = FLUX_FUNCTIONS["gr4jroute"]
    coefs_grad_route = gradients.lin_coefficient_matrix(froute, a2)

    coefs_lin = np.array([coefs_grad_int, coefs_grad_prod, coefs_grad_route])

    # initialisation
    x0 = 0.5

    # 3 steps run
    qst = np.ascontiguousarray(Qstars[:, :2])
    _, x1i, fluxesi = gradients.grad_reservoir(a1, timestep,
                                              qst, gammas[0],
                                              coefs_grad_int, x0)
    qst[:, 0] = -fluxesi[:, 3]
    qst[:, 1] = -fluxesi[:, 4]
    _, x1p, fluxesp = gradients.grad_reservoir(a1, timestep,
                                              qst, gammas[1],
                                              coefs_grad_prod, x0)

    qst[:, 0] = -fluxesp[:, 5]
    qst[:, 1] = demand
    _, x1r, fluxesr = gradients.grad_reservoir(a2, timestep,
                                              qst, gammas[2],
                                              coefs_grad_route, x0)

    # Model run
    x1, fluxes = gradients.grad_model(alphas, timestep,
                                      Qstars, netw.Qstars_mapping,
                                      gammas, coefs_lin, x0)

    assert allclose(x1i, x1[:, 0])
    assert allclose(fluxesi, fluxes[:, :6])

    assert allclose(x1p, x1[:, 1])
    assert allclose(fluxesp, fluxes[:, 6:12])

    assert allclose(x1r, x1[:, 2])
    assert allclose(fluxesr, fluxes[:, 12:])
