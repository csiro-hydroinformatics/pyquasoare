from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyquasoare import benchmarks, approx, models
from hydrodiy.io import csv

froot = Path(__file__).parent.parent

# Get production store flux functions.
# The store capacity is X1 (mm) and its filling level is S.
#
# P is the daily rainfall, E is the daily evapotranspiration.
# infiltrated rainfall : fp(S/X1) = P(1-[S/X1]^2)
# actual evapotranspiration : fe(S/X1) = E S/X1 (2-S/X1)
# percolation : fr = -2.25^4/4 (S/X1)^5
#
fluxes, _ = benchmarks.gr4jprod_fluxes_noscaling()

# Definition of interpolation points
nalphas = 20
alphas = np.linspace(0., 1.2, nalphas)

# Quadratic piecewise interpolation of the flux functions
amat, bmat, cmat, cst = approx.quad_coefficient_matrix(fluxes, alphas)

# Loading data from a test site
siteid = "203014"
fd = froot / "data" / "daily" / f"hydrodata_{siteid}.csv"
df, _ = csv.read_csv(fd, index_col=0, parse_dates=True)
nval = len(df)
time = df.index
rain = df.loc[:, "RAINFALL[mm/day]"]
evap = df.loc[:, "PET[mm/day]"]

# input accounting for interception
rain_intercept = np.maximum(rain-evap, 0.)
evap_intercept = np.maximum(evap-rain, 0.)

# Defines scaling factors applied to flux functions for
# each time step. There are 3 scalings corresponding to
# the 3 flux functions.
X1 = 500
scalings = np.column_stack([rain_intercept/X1, \
                            evap_intercept/X1, \
                            np.ones(nval)])

# Run the model using QuaSoare
s0 = 1./2
niter, s1, fx = models.quad_model(alphas, scalings, \
                                amat, bmat, cmat, s0, 1.)

sims = np.column_stack([s1*X1, fx[:, 0]*X1, \
                            -fx[:, 1]*X1, -fx[:, 2]*X1])

plt.close("all")
fig, axs = plt.subplots(nrows=4, figsize=(15, 10), layout="constrained")
for iax, ax in enumerate(axs):
    ax.plot(time, sims[:, iax])

plt.show()
