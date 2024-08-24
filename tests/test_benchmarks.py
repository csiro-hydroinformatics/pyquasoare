from pathlib import Path
from itertools import product as prod
import time
import math
import re
import pytest

import numpy as np
import pandas as pd
import scipy.integrate as sci_integrate

from hydrodiy.io import iutils

from pyrezeq import benchmarks, slow

from test_approx import generate_samples, reservoir_function

from pygme.models import gr4j

import data_reader

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent
LOGGER = iutils.get_logger("benchmarks", flog=FTEST / "test_benchmarks.log")

def test_benchmarks_error(allclose):
    siteid = data_reader.SITEIDS[0]
    df = data_reader.get_data(siteid, "daily")
    inputs = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
    X1, s0 = 1., 1./2
    with pytest.raises(ValueError, match="out of bounds"):
        gp = benchmarks.gr4jprod(1, X1, s0, inputs)

    X1, s0 = 1000., -1.
    with pytest.raises(ValueError, match="out of bounds"):
        gp = benchmarks.gr4jprod(1, X1, s0, inputs)


    inflows = inputs[:, 0]
    timestep, theta, nu, q0, s0 = 1., -1., 2., 1., 0.
    with pytest.raises(ValueError, match="out of bounds"):
        sim = benchmarks.nonlinrouting(1, timestep, theta, \
                                        nu, q0, s0, inflows)

    timestep, theta, nu, q0, s0 = 1., 1., 0.5, 1., 0.
    with pytest.raises(ValueError, match="out of bounds"):
        sim = benchmarks.nonlinrouting(1, timestep, theta, \
                                        nu, q0, s0, inflows)

    timestep, theta, nu, q0, s0 = 1., 1., 2., -1., 0.
    with pytest.raises(ValueError, match="out of bounds"):
        sim = benchmarks.nonlinrouting(1, timestep, theta, \
                                        nu, q0, s0, inflows)

    timestep, theta, nu, q0, s0 = 1., 1., 2., 1., -1.
    with pytest.raises(ValueError, match="out of bounds"):
        sim = benchmarks.nonlinrouting(1, timestep, theta, \
                                        nu, q0, s0, inflows)


def test_gr4jprod_nsubdiv(allclose):
    siteid = data_reader.SITEIDS[0]
    df = data_reader.get_data(siteid, "daily")
    inputs = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
    X1 = 500
    s0 = X1/2
    gp1 = benchmarks.gr4jprod(1, X1, s0, inputs)
    gp100 = benchmarks.gr4jprod(100, X1, s0, inputs)
    assert np.all(~np.allclose(gp1, gp100, atol=1e-5))


def test_gr4jprod_vs_gr4j(allclose):
    X1s = np.linspace(50, 1000, 100)
    mod = gr4j.GR4J()
    nsubdiv = 1 # this is the config of original gr4j model
    LOGGER.info("")

    for isite, siteid in enumerate(data_reader.SITEIDS):
        # Get climate data
        df = data_reader.get_data(siteid, "daily")
        inputs = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
        Pi = np.maximum(inputs[:, 0]-inputs[:, 1], 0)
        Ei = np.maximum(inputs[:, 1]-inputs[:, 0], 0)

        # Original gr4j model setup
        mod.allocate(inputs, noutputs=mod.noutputsmax)

        cc = ["S", "PS", "ES", "PERC"]
        errmax_max = 0.
        for X1 in X1s:
            # Run original GR4J
            mod.X1 = X1
            mod.initialise_fromdata()
            s0 = mod.states.S
            mod.run()
            sims = mod.to_dataframe()
            SR = sims.S.shift(1)/X1
            SR.iloc[0] = s0/X1

            # Compute PS and ES
            PHI = np.tanh(Pi/X1)
            PS = X1*(1-SR*SR)*PHI/(1+SR*PHI)

            PSI = np.tanh(Ei/X1)
            ES = X1*SR*(2-SR)*PSI/(1+(1-SR)*PSI)

            # Run alternative GR production with 1 subdiv
            gp = benchmarks.gr4jprod(nsubdiv, X1, s0, inputs)

            expected = np.column_stack([sims.S, PS, ES, sims.PERC, \
                                            sims.PR, sims.AE])
            err = np.abs(gp-expected)
            errmax = err.max()
            errmax_max = max(errmax, errmax_max)
            atol, rtol = 5e-5, 1e-5
            notgood = err>np.abs(expected)*rtol+atol
            assert allclose(expected, gp, atol=atol, rtol=rtol)

        mess = f"gr4jprod vs gr4j /site {isite+1}:"\
                    +f" errmax={errmax:3.3e}"
        LOGGER.info(mess)


def test_gr4jprod_vs_numerical(allclose):
    X1s = np.linspace(50, 1000, 5)
    nsubdiv = 50000
    LOGGER.info("")

    for isite, siteid in enumerate(data_reader.SITEIDS):
        df = data_reader.get_data(siteid, "daily")
        # just 2022 because it takes too much time otherwise
        inputs = df.loc["2022", ["RAINFALL[mm/day]", "PET[mm/day]"]].values

        errmax_max = 0.
        for X1 in X1s:
            s0 = X1/2
            gp = benchmarks.gr4jprod(nsubdiv, X1, s0, inputs)

            # Numerical simulation
            numerical = np.zeros((len(gp), 4))
            u_start = s0/X1
            for t in range(len(inputs)):
                # climate inputs
                P, E = inputs[t]

                # fluxes equations
                sumf, dsumf, fluxes, dfluxes = \
                            benchmarks.gr4jprod_fluxes_scaled(P, E, X1)

                # Numerical integration
                _, out, _, _ = slow.integrate_numerical(sumf, dsumf, \
                                                fluxes, dfluxes, \
                                                0, u_start, [1.])
                numerical[t, 0] = out[0]*X1
                numerical[t, 1] = out[1]*X1
                numerical[t, 2] = -out[2]*X1
                numerical[t, 3] = -out[3]*X1

                # loop
                u_start = out[0]

            assert allclose(numerical, gp[:, :4], atol=1e-4, rtol=1e-5)
            errmax = np.abs(gp[:, :4]-numerical).max()
            errmax_max = max(errmax, errmax_max)

        mess = f"gr4jprod vs num /site  errmax={errmax_max:3.3e}"
        LOGGER.info(mess)


def test_nonlinrouting_vs_analytical(allclose):
    timestep = 3600
    s0 = 0.
    LOGGER.info("")

    for isite, siteid in enumerate(data_reader.SITEIDS):
        df = data_reader.get_data(siteid, "hourly")
        df = df.loc["2022-01-01": "2022-12-31"]

        inflows = df.loc[:, "STREAMFLOW_UP[m3/sec]"].interpolate()
        q0 = inflows.quantile(0.9)
        inflows = inflows.values

        # Stored volumne in 24h
        theta = q0*24*timestep

        for nu in [1, 2]:
            # Test routing with different subdiv
            # expect different simulations
            sim1 = benchmarks.nonlinrouting(1, timestep, theta, \
                                        nu, q0, s0, inflows)
            sim100 = benchmarks.nonlinrouting(100, timestep, theta, \
                                        nu, q0, s0, inflows)
            assert np.all(~np.allclose(sim1, sim100, atol=1e-5))

            # Test routing with very small storage
            # Expect inflow==outflow
            for ns in [1, 100, 1000]:
                sim = benchmarks.nonlinrouting(ns, timestep, 1e-5, \
                                        nu, q0, s0, inflows)
                atol, rtol = (1e-6, 1e-5) if nu==1 else (1e-4, 5e-3)
                assert allclose(sim, inflows, atol=atol, rtol=rtol)

            # Simulation with high number of subdivisions
            nsubdiv = 10000
            sim = benchmarks.nonlinrouting(nsubdiv, timestep, theta,\
                                        nu, q0, s0, inflows)

            # Analytical solution
            s_start = s0
            s_start_split = s0
            expected = np.zeros_like(sim)
            split = np.zeros_like(sim)
            qi_prev = 0.
            for t in range(len(inflows)):
                qi = inflows[t]

                if nu==1:
                    # analytical linear solution
                    # ds/dt = I-q0*(s/theta)
                    # ds/(1-q0/I/theta.s) = I.dt
                    # u = q0/I/theta.s   ds = theta*I/q0.du
                    # du/(1-u) = q0/theta.Delta
                    # log((u1-1)/(u0-1)) = -q0/theta.Delta
                    # w = exp(-q0/theta.Delta)
                    # u1 = 1+w*(u0-1)
                    # s1 = I.theta/q0*(1+w*(s0.q0/I/theta-1))
                    # s1 = I.theta/q0.(1-w)+w*s0
                    w = math.exp(-q0/theta*timestep)
                    s_end = qi*theta/q0*(1-w)+w*s_start

                elif nu==2:
                    # analytical quadratic solution
                    # ds/dt = I-q0*(s/theta)^2
                    # ds/(1-(sqrt(q0/I)/theta.s)^2) = I.dt
                    # u = sqrt(q0/I)/theta.s   ds = theta/sqrt(q0/I).du
                    # du/(1-u^2) = sqrt(q0.I)/theta.Delta
                    # w = tanh(sqrt(q0.I)/theta.Delta)
                    # u1 = (u0+w)/(1+w.u0)
                    # A = sqrt(q0/I)/theta
                    # s1 = (s0+w/A)/(1+w.A.s0)
                    A = math.sqrt(q0/qi)/theta
                    w = math.tanh(math.sqrt(q0*qi)/theta*timestep)
                    s_end = (s_start+w/A)/(1+w*A*s_start)

                expected[t] = (s_start-s_end)/timestep+qi
                s_start = s_end
                qi_prev = qi

            errmax = np.abs(sim-expected).max()
            errlogmax = np.abs(np.log(sim)-np.log(expected)).max()
            mess = f"nonlin routing /site {isite+1}/nu={nu}: "\
                        f"errmax={errmax:3.3e} errlogmax={errlogmax:3.3e}"
            LOGGER.info(mess)
            assert allclose(sim, expected, atol=1e-5, rtol=1e-4)


def test_nonlinrouting_vs_numerical(allclose):
    timestep = 3600.
    nus = [1.5, 2., 5.]
    nsubdiv = 50000
    LOGGER.info("")

    for isite, siteid in enumerate(data_reader.SITEIDS):
        df = data_reader.get_data(siteid, "hourly")
        # just 2022-Feb/March because it takes too much time otherwise
        inflows = df.loc["2022-02":"2022-03", :].iloc[:, 0].values
        q0 = inflows.mean()
        theta = q0*timestep*10.

        errmax_max = 0.
        for nu in nus:
            s0 = 0.
            sim = benchmarks.nonlinrouting(nsubdiv, timestep, theta,\
                                        nu, q0, s0, inflows)

            # Numerical simulation
            numerical = np.zeros(len(sim))
            u_start = s0/theta
            for t in range(len(inflows)):
                # fluxes equations
                qin = inflows[t]
                sumf, dsumf, fluxes, dfluxes = \
                            benchmarks.nonlinrouting_fluxes_scaled(qin, q0, theta, nu)

                # Numerical integration
                _, out, _, _ = slow.integrate_numerical(sumf, dsumf, \
                                                fluxes, dfluxes, \
                                                0, u_start, [timestep])
                numerical[t] = -out[2]*theta/timestep

                # loop
                u_start = out[0]

            errmax = np.abs(sim-numerical).max()
            errmax_max = max(errmax, errmax_max)
            assert allclose(numerical, sim, atol=1e-4, rtol=1e-5)

        mess = f"nonlinrouting vs num /site {isite+1}:"\
                +f" errmax={errmax_max:3.3e}"
        LOGGER.info(mess)



