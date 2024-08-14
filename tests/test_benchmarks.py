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
LOGGER = iutils.get_logger("models", flog=FTEST / "test_models.log")

def test_gr4jprod_nsubdiv(allclose):
    mod = gr4j.GR4J()
    nsubdiv = 1 # this is the config of original gr4j model
    LOGGER.info("")

    siteid = data_reader.SITEIDS[0]
    df = data_reader.get_data(siteid, "daily")
    inputs = df.loc[:, ["RAINFALL[mm/day]", "PET[mm/day]"]].values

    X1 = 500
    s0 = X1/2
    gp1 = models.gr4jprod(1, X1, s0, inputs)
    gp100 = models.gr4jprod(100, X1, s0, inputs)
    assert np.all(~np.allclose(gp1, gp100, atol=1e-5))


def test_gr4jprod_vs_gr4j(allclose):
    X1s = np.linspace(50, 1000, 100)
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
                mess = f"gr4jprod vs gr4j /site {isite+1}/x1={X1:0.1f}:"\
                            +f" errmax={errmax:3.3e}"
                LOGGER.info(mess)

            assert allclose(gp, expected, atol=1e-10)


def test_gr4jprod_vs_numerical(allclose):
    X1s = np.linspace(50, 1000, 10)
    nsubdiv = 10000
    LOGGER.info("")

    for isite, siteid in enumerate(data_reader.SITEIDS):
        df = data_reader.get_data(siteid, "daily")
        # just 2022 because it takes too much time otherwise
        inputs = df.loc["2022", ["RAINFALL[mm/day]", "PET[mm/day]"]].values
        #inputs = inputs[55:65]

        for X1 in X1s:
            s0 = X1/2
            gp = models.gr4jprod(nsubdiv, X1, s0, inputs)

            # Numerical simulation
            numerical = np.zeros((len(gp), 4))
            u_start = s0/X1
            for i in range(len(inputs)):
                # Scaled inputs
                p, e = inputs[i]/X1
                pi = max(p-e, 0) # if>0, ae=e
                ei = max(e-p, 0) # if>0, ps=p -> pr=0

                # GR4J instantaneous equations
                nu = 1./2.25
                fpr = lambda x: pi*(1.-x**2)
                fae = lambda x: ei*x*(2.-x)
                fperc = lambda x: (nu*x)**5/4./nu
                fs = lambda x: fpr(x)-fae(x)-fperc(x)
                funs = [fs, fpr, fae, fperc]

                dfpr = lambda x: -2.*pi*x
                dfae = lambda x: 2.*ei*(1.-x)
                dfperc = lambda x: 5./4.*(nu*x)**4
                dfs = lambda x: dfpr(x)-dfae(x)-dfperc(x)
                dfuns = [dfs, dfpr, dfae, dfperc]

                _, out, _, _ = slow.integrate_forward_numerical(funs, dfuns, \
                                                0, [u_start]+[0]*3, [1.])
                numerical[i] = out*X1

                # Correct PR with interception and percolation
                numerical[i][1] = p*X1+(out[3]-out[1]+ei-e)*X1
                # Correct AE with interception
                numerical[i][2] += (p-pi)*X1

                # loop
                u_start = out[0]

            #import matplotlib.pyplot as plt
            #from hydrodiy.plot import putils
            #fig, axs = plt.subplots(ncols=2, nrows=2, \
            #                            figsize=(15, 12), layout="tight")
            #names = ["S", "PR", "AE", "PERC"]
            #for i, ax in enumerate(axs.flat):
            #    n, s1, s1000 = numerical[:, i], gp1[:, i], gp1000[:, i]
            #    ax.plot(n, "k-", label="numerical")
            #    #ax.plot(s1, "b--", label="split 1")
            #    ax.plot(s1000, "b:", label="split 1000")
            #    #if names[i] == "AE":
            #    #    ax.plot(inputs[:, 0], "b:", label="rain")
            #    #    ax.plot(inputs[:, 1], "g:", label="pet")
            #    ax.set(title=names[i])
            #    ax.legend(loc=2, ncol=2)

            #    tax = ax.twinx()
            #    #tax.plot(s1-n, "r--", alpha=0.3)
            #    tax.plot(s1000-n, "r:", alpha=0.3)

            #    ymax = 1e-2
            #    tax.set(ylim=(-ymax, ymax))
            #    putils.line(tax, 1, 0, 0, 0, "-", lw=0.9, color="pink")

            #plt.show()
            #import pdb; pdb.set_trace()

            errmax = np.abs(gp-numerical).max()
            if X1 == X1s[len(X1s)//2]:
                mess = f"gr4jprod vs num /site {isite+1}/x1={X1:0.1f}:"\
                            +f" errmax={errmax:3.3e}"
                LOGGER.info(mess)

            assert allclose(numerical, gp, atol=1e-3, rtol=1e-4)



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
            errlogmax = np.abs(np.log(sim)-np.log(expected)).max()
            mess = f"nonlin routing /site {isite+1}/nu={nu}: "\
                        f"errmax={errmax:3.3e} errlogmax={errlogmax:3.3e}"
            LOGGER.info(mess)
            assert allclose(sim, expected, atol=1e-5, rtol=1e-4)

