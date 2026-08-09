#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2026-08-03 14:59:50.957360
## Comment : Approximation of integration
##
## ------------------------------


import sys
import os
import re
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from hydrodiy.io import csv, iutils

from pyquasoare import approx, integrate, slow, benchmarks, steady

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------

verif = True

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent.parent.parent

# ----------------------------------------------------------------------
# @Logging
# ----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------
f = lambda x: approx.quad_fun(coefs, x)
df = lambda x: approx.quad_grad(coefs, x)

for icase in [7]: #range(7):
    if icase == 0:
        a=-4.887e+00;b=-3.148e+00;c=9.709e-01;s0=-5.526e-05;t0=5.000e-01;t1=5.000e+00
    elif icase == 1:
        a=-6.159e+02;b=-4.992e+03;c=3.594e+03;s0=-2.382e+00;t0=5.000e-01;t1=5.000e+00
    elif icase == 2:
        a=1.620e-09;b=0.503e+00;c=-3.568e+00;s0=4.853e+00;t0=5.000e-01;t1=5.000e+00
    elif icase == 3:
        a=0.000e+00;b=3.449e-09;c=2.000e+00;s0=1.602e+00;t0=5.000e-01;t1=5.000e+00
    elif icase == 4:
        a=0.000e+00;b=4.960e-09;c=-3.636e+00;s0=-3.009e+00;t0=5.000e-01;t1=5.000e+00
    elif icase == 5:
        a=3.572e-09;b=-3.752e-01;c=-8.034e-01;s0=2.538e+00;t0=5.000e-01;t1=5.000e+00
    elif icase == 6:
        a=2.027e-09;b=8.279e-01;c=-1.476e+00;s0=3.696e+00;t0=5.000e-01;t1=1.000e+00
    elif icase == 7:
        a=-1.6564e+05;b=2.7878e+05;c=-1.1580e+05;t0=1.0942e-05;t1=1.0000e+00;s0=9.5745e-01;

    avect = np.array([1., 0., a - 1.])
    bvect = np.array([0., 1., b - 1.])
    cvect = np.array([0., 0., c])

    Delta, qD, sbar = integrate.quad_constants(a, b, c)
    s1 = integrate.quad_forward(a, b, c, Delta, qD, sbar, t0, s0, t1)

    fluxes = np.zeros(len(avect))
    integrate.quad_fluxes_increment(avect, bvect, cvect,
                    a, b, c, Delta, qD, sbar, t0, t1, s0, s1, fluxes)

    # ODE solver
    nfluxes = len(avect)
    v = np.zeros(nfluxes + 1)
    def fun_ivp(t, y):
        total = 0.
        for i in range(nfluxes):
            s = y[-1]
            f = avect[i] * s**2 + bvect[i] * s + cvect[i]
            total += f
            v[i] = f
        v[-1] = total
        return v

    m = np.zeros((nfluxes + 1, nfluxes + 1))
    def jac_ivp(t, y):
        total = 0.
        for i in range(nfluxes):
            s = y[-1]
            df = 2 * avect[i] + bvect[i]
            total += df
            m[i, 0] = df
        m[nfluxes, 0] = total
        return m

    y0 = [0.] * 3 + [s0]
    t = [t0, t1]
    res = solve_ivp(fun=fun_ivp,
                    t_span=[t0, t[-1]],
                    y0=np.array([0.] * nfluxes + [s0]),
                    method="Radau",
                    max_step=np.inf,
                    jac=jac_ivp,
                    t_eval=t)
    fluxes_expected = res.y[:nfluxes, -1]
    s1_expected = res.y[-1, -1]

    tt = np.linspace(t0, t1, 100000)
    s1_t = np.array([integrate.quad_forward(a, b, c, Delta,
                                            qD, sbar, t0, s0, t)
                                            for t in tt])
    sum_li = (s1_t[1:] * np.diff(tt)).sum()
    err = abs(sum_li - fluxes_expected[1]) / fluxes_expected[1]
    tol = 8e-4
    assert err < tol, f"err li = {err:0.3e}"

    sum_sq = (s1_t[1:]**2 * np.diff(tt)).sum()
    err = abs(sum_sq - fluxes_expected[0]) / fluxes_expected[0]
    assert err < tol, f"err sq = {err:0.3e}"

    print(f"\n-- Case {icase} ---")
    print(f"s1 ode : {s1_expected:10.3e}")
    print(f"s1 quas: {s1:10.3e} (err={s1 - s1_expected:10.3e})")

    toprint = lambda x: "  ".join([f"integS{'2' if iu == 0 else ''}={u:10.2e}"
                                  for iu, u in enumerate(x)])
    fluxes_expected = fluxes_expected[:2]
    print(f"fx ode : {toprint(fluxes_expected)}")
    fluxes = fluxes[:2]
    print(f"fx quas: {toprint(fluxes)}")


LOGGER.completed()

