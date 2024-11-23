from pathlib import Path
import sys
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyquasoare import benchmarks, approx, models
from hydrodiy.io import csv


#
# Code to solve the reservoir equation associated
# with the VIC soil moisture model as per Kavetski
# et al. (2006):
#
# dS/dt = P*(1-S/Smax)^alpha -Kb(S/Smax)^beta
#            -E[1-(1-S/Smax)^gamma]
#
# where P is rain and E is PET.
#
# In this implementation, the model is re-written as
#
# du/dt =   P/Smax
#            - P/Smax  * [1-(1-u)^alpha]
#            - Kb/Smax * u^beta
#            - E/Smax  * [1-(1-u)^gamma]
#
# where u = S/Smax
#
# Introducing the flux scaling factors, we finally obtain
#
# du/dt =  s1*f1(u)     -> f1(u) = 1 (constant)    s1=P/Smax
#          + s2*f2(u)   -> f2(u) = -1+(1-u)^alpha  s2=P/Smax
#          + s3*f3(u)   -> f3(u) = -u^beta         s3=Kb/Smax
#          + s4*f4(u)   -> f4(u) = -1+(1-u)^gamma  s4=E/Smax
#
# Kavetski, D., G. Kuczera, and S. W. Franks (2006),
# Bayesian analysis of input uncertainty in hydrological
# modeling: 2. Application, Water Resour. Res., 42,
# W03408, doi:10.1029/2005WR004376.
#

source_file = Path(__file__).resolve()
froot = source_file.parent
basename = source_file.stem
fimg = froot / "images"
fimg.mkdir(exist_ok=True, parents=True)
for f in fimg.glob("*.png"):
    f.unlink()

# Definition of interpolation points
# magnifying the origin and u=1 due to strong non-linearities
nalphas = 12
e = np.linspace(0, 1, nalphas-2)
alphas = np.concatenate([[-0.01], (np.sin((e-0.5)*math.pi)+1)/2, \
                            [1.01]])

# Loading data from a test site
siteid = "203014"
fd = froot.parent.parent / "data" / "daily" / f"hydrodata_{siteid}.csv"
df, _ = csv.read_csv(fd, index_col=0, parse_dates=True)
nval = len(df)
day = df.index
rain = df.loc[:, "RAINFALL[mm/day]"].values
evap = df.loc[:, "PET[mm/day]"].values

# Generates random parameter samples
nsamples = 5
pnames = ["logSmax", "logkb", "alpha", "beta", "logGamma"]
pmin = np.array([1, -1., 0., 0., -1.])[None, :]
pmax = np.array([2., 1., 1., 2., 1.])[None, :]
nparams = len(pnames)
params = np.random.uniform(0, 1, size=(nsamples, nparams))*(pmax-pmin)+pmin
params = pd.DataFrame(params, columns=pnames)

# Run model for each parameter set
tot = 0
for iparam, param in params.iterrows():
    print(f"Running parameter {iparam+1}/{nsamples}")
    logSmax, logKb, alpha, beta, logGamma = param.values
    Smax = 10**logSmax
    kb = 10**logKb
    gamma = 10**logGamma

    # Flux functions
    # .. rainfall inputs
    f1 = lambda u: 1.
    # .. Effective rainfall
    mm = lambda x: min(1, max(0, x))
    f2 = lambda u: -1+(1-mm(u))**alpha
    # .. Baseflow
    f3 = lambda u: -mm(u)**beta
    # .. Actual ET
    f4 = lambda u: -1+(1-mm(u))**gamma

    fluxes = [f1, f2, f3, f4]

    # Defines scaling factors for each flux
    scalings = np.column_stack([rain/Smax, rain/Smax, \
                                    np.ones(nval)*kb/Smax, evap/Smax])

    # Run the model using QuaSoare
    # .. Quadratic piecewise interpolation of the flux functions
    start = time.time()
    amat, bmat, cmat, cst = approx.quad_coefficient_matrix(fluxes, alphas)

    # .. Ode solver
    u0 = 1./2
    niter, u1, fx = models.quad_model(alphas, scalings, \
                                    amat, bmat, cmat, u0, 1.)
    end = time.time()
    tot += end-start

    # Convert simulated flux
    sims = np.column_stack([u1, -fx[:, 1]*Smax, \
                                -fx[:, 2]*Smax, -fx[:, 3]*Smax])
    sims = pd.DataFrame(sims, columns=["Store", "Peff", "Bflow", "AET"], \
                            index=day)

    # Plots
    # .. Check interpolation
    xx = np.linspace(0, 1, 500)
    yhats = approx.quad_fun_from_matrix(alphas, amat, bmat, cmat, xx)

    plt.close("all")
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5), layout="constrained")
    for iax, ax in enumerate(axs):
        ytrue = [fluxes[iax+1](x) for x in xx]
        ax.plot(xx, ytrue, "k-", label="Original")
        ax.plot(xx, yhats[:, iax+1], "k--", label="Interpolated")
        ax.plot(alphas, [fluxes[iax+1](a) for a in alphas], "ro", label="nodes")
        ax.set(title=f"Function {iax+1}")
        ax.legend()

    ftitle = f"Smax={Smax:0.1e} Kb={kb:0.1e} alpha={alpha:0.1e} beta={beta:0.1e} gamma={gamma:0.1e}"
    fig.suptitle(ftitle, fontweight="bold")
    fp = fimg / f"VIC_param{iparam+1}_fluxes.png"
    fig.savefig(fp)

    # .. simulations
    plt.close("all")
    fig, axs = plt.subplots(nrows=sims.shape[1], figsize=(15, 10),\
                                layout="constrained", sharex=True)
    for iax, ax in enumerate(axs):
        se = sims.iloc[:, iax]
        sims.iloc[:, iax].plot(ax=ax, legend=False)
        ax.set_title(se.name, x=0.01, y=0.92, va="top", \
                ha="left", fontsize="large", fontweight="bold")

    fig.suptitle(ftitle, fontweight="bold")
    fp = fimg / f"VIC_param{iparam+1}_sim.png"
    fig.savefig(fp)

print(f"runtime = {tot/nsamples/nval*3650:0.2e} sec/10 years of daily simulation")
print("Process completed")
