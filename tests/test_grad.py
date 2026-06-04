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
import c_pyquasoare

import data_reader

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent
LOGGER = iutils.get_logger("models", flog=FTEST / "test_grad.log")

def fixed(x, x0):
    return x0 * np.ones_like(x)

def indicator(x, xmin, xmax):
    return np.where((x > xmin) & (x < xmax), 1., 0.)


RESERVOIR_FUNCTIONS = {
    "gr4jprod": [
        lambda x: 1 - np.minimum(x, 1)**2,
        lambda x: np.minimum(x, 1) * (2 - np.minimum(x, 1)),
        lambda x: 4**3/9**4 * x**5
        ],
    "gr4jprodquad": [
        lambda x: 1 - np.minimum(x, 1)**2,
        lambda x: np.minimum(x, 1) * (2 - np.minimum(x, 1)),
        lambda x: 4**3/9**4 * x**2
        ],
    "gr4jroute": [
        lambda x: 0.9 * fixed(x, 1.),
        lambda x: x ** 3.5,
        lambda x: x**5 / 4
        ],
    "gr4jroutequad": [
        lambda x: 0.9 * fixed(x, 1.),
        lambda x: x ** 2,
        lambda x: x**2 / 4
        ],
    "quadrouting": [
        lambda x: indicator(x, 0., np.inf),
        lambda x: fixed(x, 0.),
        lambda x: x**2
        ],
    "nonlinrouting": [
        lambda x: indicator(x, 0., np.inf),
        lambda x: fixed(x, 0.),
        lambda x: x**6
        ],
    "xinanjiang": [
        lambda x: (1 - np.minimum(x, 1.))**0.2,
        lambda x: indicator(x, 0., np.inf),
        lambda x: fixed(x, 0.)
        ],
    "intercept": [
        lambda x: indicator(x, 0., 1.),
        lambda x: indicator(x, 0., np.inf),
        lambda x: fixed(x, 0.)
        ],
    "muskingumrouting": [
        lambda x: indicator(x, 0., np.inf),
        lambda x: fixed(x, 0.),
        lambda x: x**6
        ],
}

def test_network():
    netw = gradients.Network(3)
    netw.add_link_total_outflow(0, 1)
    netw.add_link_total_outflow(1, 2)
    assert netw.upstream_reservoirs(2) == {0, 1}

    # This is cyclical -> error
    with pytest.raises(ValueError, match="cyclical"):
        netw.add_link_total_outflow(2, 0)

    netw = gradients.Network(3)
    netw.add_link_bypass_shortfall(0, 1)
    netw.add_link_bypass_shortfall(1, 2)
    assert netw.upstream_reservoirs(2) == {0, 1}

    # This is cyclical -> error
    with pytest.raises(ValueError, match="cyclical"):
        netw.add_link_bypass_shortfall(2, 0)



@pytest.mark.parametrize("reservoir", RESERVOIR_FUNCTIONS)
def test_lin_coefficient_matrix(reservoir, allclose):
    alphas = np.linspace(0, 1, 50)
    funs = RESERVOIR_FUNCTIONS[reservoir]
    coefs = gradients.lin_coefficient_matrix(funs, alphas)

    for i, f in enumerate(funs):
        y1 = f(alphas)[:-1]
        y2 = coefs[i][:, 0] * alphas[:-1] + coefs[i][:, 1]
        assert allclose(y1, y2)



def test_grad_integrate_error():
    # Lin
    n = np.zeros(1, dtype=np.int32)
    x = np.zeros(1, dtype=np.float64)
    fx = np.zeros(6, dtype=np.float64)

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

    qst = -np.ones(2)
    ierr = c_pyquasoare.grad_integrate(alphas, timestep, qst,
                                      gamma, coefs_lin,
                                      0, x_start,
                                      n, x, fx)
    assert ierr == 112

    qst = np.ones(2)
    coefs_lin[0] *= -1
    ierr = c_pyquasoare.grad_integrate(alphas, timestep, qst,
                                      gamma, coefs_lin,
                                      0, x_start,
                                      n, x, fx)
    assert ierr == 111


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", RESERVOIR_FUNCTIONS)
@pytest.mark.parametrize("gamma", [10, 100, 10000])
@pytest.mark.parametrize("timestep", [0.1, 1])
def test_grad_integrate(siteid, reservoir, gamma, timestep, allclose):
    # Get data
    df = data_reader.get_data(siteid, "daily").iloc[:100]
    Qstars = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
    Qstars[:, 1] *= -1 # PET = negative demand
    nval = len(Qstars)

    # Get coefs - quad coefs with option = 0 (linear)
    nalphas = 20
    alphas = np.linspace(0, 1, nalphas)
    funs = RESERVOIR_FUNCTIONS[reservoir]
    coefs = approx.quad_coefficient_matrix(funs, alphas,
                                           approx_opt=0)
    coefs_lin = gradients.lin_coefficient_matrix(funs, alphas)

    # Loop and integrate
    x_start = 0.5
    scalings = np.zeros(3)
    for i in range(nval):
        # Quad interpol
        scalings[0] = Qstars[i, 0] / gamma
        scalings[1] = Qstars[i, 1] / gamma
        scalings[2] = - 1. / timestep
        n1, x1, fx1 = integrate.quad_integrate(alphas, scalings,
                                               coefs, 0.,
                                               x_start, timestep)
        fx1 *= gamma / timestep

        # Lin
        n2 = np.zeros(1, dtype=np.int32)
        x2 = np.zeros(1, dtype=np.float64)
        fx2 = np.zeros(6, dtype=np.float64)
        qst = np.ascontiguousarray(Qstars[i])
        ierr = c_pyquasoare.grad_integrate(alphas, timestep, qst,
                                          gamma, coefs_lin,
                                          0, x_start,
                                          n2, x2, fx2)
        assert allclose(x1, x2[0])
        assert allclose(fx1, fx2[:3])

        fxa = np.array([fx1[0] - qst[0], fx1[1] - qst[1], fx1[0] - qst[0] + fx1[2]])
        assert allclose(fxa, fx2[3:])

        x_start = x1


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("reservoir", RESERVOIR_FUNCTIONS)
@pytest.mark.parametrize("gamma", [1, 10, 100, 1000])
@pytest.mark.parametrize("timestep", [0.1, 1])
def test_grad_reservoir(siteid, reservoir, gamma, timestep, allclose):
    # Get data
    df = data_reader.get_data(siteid, "daily")
    Qstars = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
    Qstars[:, 1] *= -1 # PET = negative demand
    Qstars = np.ascontiguousarray(Qstars)
    nval = len(Qstars)

    scalings = np.ones((nval, 3))
    scalings[:, :2] = Qstars / gamma
    scalings[:, 2] = - 1. / timestep

    # Get coefs - quad coefs with option = 0 (linear)
    nalphas = 20
    alphas = np.linspace(0, 1, nalphas)
    funs = RESERVOIR_FUNCTIONS[reservoir]
    coefs = approx.quad_coefficient_matrix(funs, alphas,
                                           approx_opt=0)
    stdy = steady.quad_steady_flux_scalings(alphas, coefs, scalings)
    amax = max(1.1, np.nanmax(stdy))
    alphas = np.linspace(0, amax, nalphas)

    coefs = approx.quad_coefficient_matrix(funs, alphas,
                                           approx_opt=0)
    coefs_lin = gradients.lin_coefficient_matrix(funs, alphas)

    # Quadratic
    x0 = 0.5
    n1, x1, fx1 = models.quad_model(alphas, scalings, coefs,
                                    x0, timestep)
    fx1 *= gamma / timestep

    # Lin
    n2, x2, fx2 = gradients.grad_reservoir(alphas, timestep,
                                          Qstars, gamma, coefs_lin, x0)
    fx1 = np.column_stack([fx1,
                           -Qstars[:, 0] + fx1[:, 0],
                           -Qstars[:, 1] + fx1[:, 1],
                           -Qstars[:, 0] + fx1[:, 0] + fx1[:, 2]])
    assert allclose(x1, x2)
    assert allclose(fx1, fx2)

    # Sanity check
    if not re.search("rout(e|ing)", reservoir):
        assert all((x2 >= 0) & (x2 <= 1.05))

    # Mass balance
    dSa = np.diff(x2) * gamma
    dSb = timestep * fx2[1:, :3].sum(axis=1)
    assert allclose(dSa, dSb)


@pytest.mark.parametrize("siteid", data_reader.SITEIDS)
@pytest.mark.parametrize("timestep", [0.1, 1])
def test_grad_gr4j(siteid, timestep, allclose):
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

    fint = RESERVOIR_FUNCTIONS["intercept"]
    coefs_grad_int = gradients.lin_coefficient_matrix(fint, a1)

    fprod = RESERVOIR_FUNCTIONS["gr4jprod"]
    coefs_grad_prod = gradients.lin_coefficient_matrix(fprod, a1)

    froute = RESERVOIR_FUNCTIONS["gr4jroute"]
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

