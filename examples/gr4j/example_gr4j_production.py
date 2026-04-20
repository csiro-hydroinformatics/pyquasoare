from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pyquasoare import approx, models
from hydrodiy.io import csv
#
# Code to solve the reservoir equation associated
# with the GR4J production reservoir per Perring
# et al. (2006):
#
# dS/dt = P * [1 - (S/X1)^2] - E * S / X1 * (2 - S/X1) - 1/2.25^5/4 * u^5
#
# where P is rain and E is PET and X1 is the
# reservoir capacity.
#
# In this implementation, the model is re-written as
#
# du/dt =  P/X1 * (1 - u^2)
#          - E/X1  * u * (2 - u)
#          - 1 / 2.25^5 / X1 / 4 * u^5
#
# where u = S/X1
#
# Introducing the flux scaling factors, we finally obtain
#
# du/dt =  sc1 * f1(u)   -> f1(u) = 1 -u^2          sc1 = P / X1
#        + sc2 * f2(u)   -> f2(u) = -u * (2 - u)    sc2 = E / X1
#        + sc3 * f3(u)   -> f3(u) = -u^5            sc3 = 1 / 2.25^4 / 4 / X1
#

# Flux functions:
# .. infiltrated rainfall
def f1(u):
    return 1 - u**2

# .. actual ET
def f2(u):
    return -u * (2 - u)

# .. percolation
def f3(u):
    return - u**5

fluxes = [f1, f2, f3]

# Definition of interpolation points
nalphas = 20
alphas = np.linspace(0., 1.2, nalphas)

# Quadratic piecewise interpolation of the flux functions
coefs = approx.quad_coefficient_matrix(fluxes, alphas)

# Loading data from a test site
siteid = "203014"
source_file = Path(__file__).resolve()
froot = source_file.parent
fd = froot.parent.parent / "data" / "daily" / f"hydrodata_{siteid}.csv"
df, _ = csv.read_csv(fd, index_col=0, parse_dates=True)
nval = len(df)
time = df.index
rain = df.loc[:, "RAINFALL[mm/day]"]
evap = df.loc[:, "PET[mm/day]"]

# Remove interception from inputs similarly to what is
# done in GR4J
rain_intercept = np.maximum(rain - evap, 0.)
evap_intercept = np.maximum(evap - rain, 0.)

# Defines scaling factors applied to flux functions for
# each time step. There are 3 scalings corresponding to
# the 3 flux functions.
X1 = 500
scalings = np.column_stack([rain_intercept / X1,
                            evap_intercept / X1,
                            (1 / 2.25**4 / X1 / 4) * np.ones(nval)])

# Run the model using QuaSoare
s0 = 1. / 2
niter, s1, fx = models.quad_model(alphas, scalings,
                                  coefs, s0, 1.)

sims = np.column_stack([s1 * X1,
                        fx[:, 0] * X1,
                        -fx[:, 1] * X1,
                        -fx[:, 2] * X1])

plt.close("all")
fig, axs = plt.subplots(nrows=4, figsize=(15, 10), layout="constrained")
names = ["Store level", "Infiltrated rainfall",
         "Actual Evapotranspiration",
         "Percolation"]
for iax, ax in enumerate(axs):
    ax.plot(time, sims[:, iax])
    ax.set(title=names[iax])

basename = source_file.stem
fimg = froot / "images"
fimg.mkdir(exist_ok=True, parents=True)
fp = fimg / f"{basename}.png"
fig.savefig(fp)
