#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : ler015
## Created : 2025-02-17 15:37:00.731309
## Comment : Verification of QuaSoARe anaytical solution
##
## ------------------------------


import sys
import math
from pathlib import Path
from string import ascii_letters as letters

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyquasoare import integrate, steady

np.random.seed(5446)

# ----------------------------------------------------------------------
# @Config
# ----------------------------------------------------------------------
param_min = -2
param_max = 2

ntimesteps = 20000

t0 = 0.

# Number of parameter configs
nrepeat = 20

# Configure plot
ncols = 4
awidth = 4
aheight = 4
fdpi = 200

# ----------------------------------------------------------------------
# @Folders
# ----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent

fimg = froot

# ----------------------------------------------------------------------
# @Process
# ----------------------------------------------------------------------

# Formula directly derived from the QuaSoARe paper, Table 3
def forward(a, b, c, s0, t):
    D = b**2 - 4 * a * c
    sD = -1 if D < 0 else 1
    qD = math.sqrt(abs(D)) / 2
    w = np.tan(qD*t) if D < 0 else np.tanh(qD*t)
    sb = -b/2/a
    return sb+(s0-sb-sD*qD/a*w)/(1-a*(s0-sb)/qD*w)

# Initialise plot
plt.close("all")
nrows = nrepeat // ncols
mosaic = np.array_split(np.arange(nrepeat), nrows)
ncols = max([len(m) for m in mosaic])
mosaic = [list(m) + ["."] * (ncols - len(m)) for m in mosaic]

plt.close("all")
fig = plt.figure(figsize=(ncols * awidth, nrows * aheight),
                 layout="constrained")
kw = dict(wspace=0.05, hspace=0.05)
axs = fig.subplot_mosaic(mosaic, gridspec_kw=kw)

# Repeat
nax = 0
while nax < nrepeat:
    a, b, c = np.random.uniform(param_min, param_max, 3)

    # Set s0 inside stable region to avoid getting
    # inifinite results
    x0, x1 = steady.quad_steady(a, b, c)
    if np.isnan(x0) or np.isnan(x1):
        continue

    dx = x1 - x0
    if a > 0:
        s0 = np.random.uniform(x0 - 3 * dx, x0)
    else:
        s0 = np.random.uniform(x0, x0 + 3 * dx)

    # Set final time
    sinf = x0
    f0 = a * s0**2 + b * s0 + c
    finf = a * sinf**2 + b * sinf + c
    tinf = abs((x0 - s0) * 2. / (finf + f0)) * 1.5
    t = np.linspace(t0, tinf, ntimesteps)

    # Run integration form QuaSoARe code
    Delta, qD, sbar = integrate.quad_constants(a, b, c)
    s_code = integrate.quad_forward(a, b, c, Delta, qD, sbar, t0, s0, t)

    # Run integration from paper formula
    s_paper = forward(a, b, c, s0, t)

    # numerical derivation to check derivative respect ODE
    dt = t[1] - t[0]
    ds_code = np.diff(s_code) / dt
    ds_theory = a * s_code[1:]**2 + b * s_code[1:] + c
    ds_paper = np.diff(s_paper) / dt

    # Plot
    # .. time series
    ax = axs[nax]
    ax.plot(t, s_code, lw=3, color="tab:blue", label="QuaSoARe code")
    ax.plot(t, s_paper, color="tab:orange", label="QuaSoARe paper")
    ax.legend(loc=2, fontsize="small", framealpha=0.)

    # .. derivative
    axi = ax.inset_axes((0.4, 0.2, 0.55, 0.35))
    axi.plot(s_code[1:], ds_theory, lw=3,
             label=r"Theory $\frac{dS}{dt} = a^2 S+bS+c$",
             color="tab:pink")
    axi.plot(s_code[1:], ds_code, label="QuaSoARe code")
    axi.legend(loc=1, fontsize="x-small", framealpha=0.)
    axi.text(0.5, 0.02, "S(t)", va="bottom", ha="center",
             transform=axi.transAxes)
    axi.text(0.02, 0.5, r"$\frac{dS}{dt}$", va="center", ha="left",
             transform=axi.transAxes)
    axi.set(xticks=[], yticks=[])

    title = f"({letters[nax]}) Simulation {nax}\n"\
            + f" a={a:0.2f}"\
            + f" b={b:0.2f}"\
            + f" c={c:0.2f}"\
            + f" $\Delta = {b**2-4*a*c:0.2f}$"
    xlabel = "Time $t$"
    ylabel = "Storage $S(t)$" if nax % ncols == 0 else ""
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    nax += 1

fp = fimg / f"verification.png"
fig.savefig(fp, dpi=fdpi)
