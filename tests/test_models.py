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

from pyrezeq import models, slow

from test_approx import generate_samples, reservoir_function

from pygme.models import gr4j

import data_reader

np.random.seed(5446)

source_file = Path(__file__).resolve()
FTEST = source_file.parent
LOGGER = iutils.get_logger("integrate", flog=FTEST / "test_integrate.log")


def test_gr4jprod_vs_gr4j(allclose):
    X1s = np.linspace(10, 1000, 100)
    mod = gr4j.GR4J()
    nsubdiv = 1 # this is the config of original gr4j model
    LOGGER.info("")

    for isite, siteid in enumerate(data_reader.SITEIDS):
        df = data_reader.get_data(siteid, "daily")
        inputs = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values
        mod.allocate(inputs, noutputs=mod.noutputsmax)

        cc = ["S", "PR", "AE", "PERC"]
        for X1 in X1s:
            # Run original GR4J
            mod.X1 = X1
            mod.initialise_fromdata()
            s0 = mod.states.S
            mod.run()
            expected = mod.to_dataframe().loc[:, cc].values

            # Run alternative GR production with 1 subdiv
            gp = models.gr4jprod(nsubdiv, X1, s0, inputs)

            errmax = np.abs(gp-expected).max()
            if X1 == X1s[len(X1s)//2]:
                mess = f"gr4jprod for site {isite+1}/x1={X1:0.1f}: errmax={errmax:3.3e}"
                LOGGER.info(mess)

            assert allclose(gp, expected, atol=5e-4, rtol=1e-4)


def test_gr4jprod_vs_numerical(allclose):
    X1s = np.linspace(10, 1000, 10)
    nsubdiv = 1000

    for siteid in data_reader.SITEIDS:
        df = data_reader.get_data(siteid, "daily")
        # just 2022 because it takes too much time otherwise
        inputs = df.loc["2022", ["RAINFALL[mm/day]", "PET[mm/day]"]].values

        for X1 in X1s:
            s0 = X1/2
            gp = models.gr4jprod(nsubdiv, X1, s0, inputs)

            # Numerical simulation
            numerical = np.zeros(len(gp))
            s_start = s0
            for i in range(len(inputs)):
                p, e = inputs[i]/X1
                funs = [lambda x: p*(1.-x**2)-e*x*(2.-x)-(4./9.*x)**5/4.]
                dfuns = [lambda x: -2.*p*x-2.*e*(1.-x)-4./9.*(4./9.*x)**4]

                _, s_end, _, _ = slow.integrate_forward_numerical(funs, dfuns, \
                                                0, [s_start], [1.])

                numerical[i] = s_end
                s_start = s_end

            assert allclose(gp[:, 0], numerical, atol=5e-4, rtol=1e-4)



def test_nonlinrouting_vs_analytical(allclose):
    delta = 3600
    s0 = 0.
    LOGGER.info("")

    for isite, siteid in enumerate(data_reader.SITEIDS):
        df = data_reader.get_data(siteid, "hourly")
        df = df.loc["2022-01-01": "2022-12-31"]

        inflows = df.loc[:, "STREAMFLOW_UP[m3/sec]"].interpolate()
        q0 = inflows.quantile(0.9)
        inflows = inflows.values

        # Stored volumne in 24h
        theta = q0*24*delta

        for nu in [1, 2]:
            # Test routing with different subdiv
            # expect different simulations
            sim1 = models.nonlinrouting(1, delta, theta, \
                                        nu, q0, s0, inflows)
            sim100 = models.nonlinrouting(100, delta, theta, \
                                        nu, q0, s0, inflows)
            assert np.all(~np.allclose(sim1, sim100, atol=1e-5))

            # Test routing with very small storage
            # Expect inflow==outflow
            for ns in [1, 100, 1000]:
                sim = models.nonlinrouting(ns, delta, 1e-5, \
                                        nu, q0, s0, inflows)
                atol, rtol = (1e-6, 1e-5) if nu==1 else (1e-4, 5e-3)
                assert allclose(sim, inflows, atol=atol, rtol=rtol)

            # Simulation with high number of subdivisions
            nsubdiv = 10000
            sim = models.nonlinrouting(nsubdiv, delta, theta,\
                                        nu, q0, s0, inflows)

            # Analytical solution
            s_start = s0
            s_start_split = s0
            expected = np.zeros_like(sim)
            split = np.zeros_like(sim)
            qi_prev = 0.
            for i in range(len(inflows)):
                qi = inflows[i]

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
                    w = math.exp(-q0/theta*delta)
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
                    w = math.tanh(math.sqrt(q0*qi)/theta*delta)
                    s_end = (s_start+w/A)/(1+w*A*s_start)

                expected[i] = (s_start-s_end)/delta+qi
                s_start = s_end
                qi_prev = qi

            errmax = np.abs(sim-expected).max()
            mess = f"nonlin routing for site {isite+1}/nu={nu}: errmax={errmax:3.3e}"
            LOGGER.info(mess)
            assert allclose(sim, expected, atol=1e-5, rtol=1e-4)

