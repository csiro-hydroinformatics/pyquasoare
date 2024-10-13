from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyquasoare import benchmarks, approx, models
from hydrodiy.io import csv

source_file = Path(__file__).resolve()
froot = source_file.parent.parent
basename = source_file.stem
fimg = froot / "examples" / "images"
fimg.mkdir(exist_ok=True, parents=True)

# Get production store flux functions.
# The store capacity is X1 (mm) and its filling level is S.
# The equation is solved using normalised fluxes u = S/X1
#
# P is the daily rainfall, E is the daily evapotranspiration.
# infiltrated rainfall : fp(u) = P/X1.(1-u^2)
# actual evapotranspiration : fe(u) = E/X1.u(2-u)
# percolation : fr = -2.25^4/4 u^5
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

# Remove interception from inputs similarly to what is
# done in GR4J
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
names = ["Store level", "Intercepted Rain", "Actual Evapotranspiration", \
                "Percolation"]
for iax, ax in enumerate(axs):
    ax.plot(time, sims[:, iax])
    ax.set(title=names[iax])

fp = fimg / f"{basename}.png"
fig.savefig(fp)
