# pyquasoare
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10065353.svg)](https://doi.org/10.5281/zenodo.10065353) [![CI](https://github.com/csiro-hydroinformatics/pyquasoare/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/csiro-hydroinformatics/pyquasoare/actions/workflows/python-package-conda.yml) [![codecov](https://codecov.io/gh/csiro-hydroinformatics/pyquasoare/graph/badge.svg?token=ARBFW69TI3)](https://codecov.io/gh/csiro-hydroinformatics/pyquasoare)

Python and C package to solve the reservoir differential equation using a
piecewise quadratic interpolation following the QuaSoARe method.

# What is pyquasoare?
This package implements the Quadratic Solution of the Approximate Reservoir 
Equation (QuaSoARe) method described in the following paper:
Lerat, J. (2024),  
"Quadratic solution of the approximate reservoir equation (QUASOARE)", HESS, Submitted.

# Installation
- Create a suitable python environment. We recommend using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) combined with the environment specification provided in the [env\_mini2.yml] (env_mini2.yml) file in this repository.
- Git clone this repository and run `pip install .`

# Basic use
Solution of the production store from the [![GR4J](https://www.sciencedirect.com/science/article/pii/S0022169403002257)] daily rainfall-runoff model using QuaSoAre:

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyquasoare import benchmarks, approx, models
from hydrodiy.io import csv

# Package root path (might need modification)
froot = Path(__file__).parent.parent

# Get flux functions for the GR4J production store
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
df = data_reader.get_data(siteid, "daily")
df = df.loc[start:end]
nval = len(df)
rain = df.loc[:, "RAINFALL[mm/day]"]
evap = df.loc[:, "PET[mm/day]"]

# input accounting for interception
rain_intercept = np.maximum(rain-evap, 0.)
evap_intercept = np.maximum(evap-rain, 0.)

# Defines scaling factors applied to flux functions for
# each time step. There are 3 scalings corresponding to 
# the 3 flux functions. 
scalings = np.column_stack([rain_intercept/X1, \
                            evap_intercept/X1, \
                            np.ones(nval)])

# Run the model using QuaSoare
X1 = 500
s0 = 1./2
niter, s1, fx = models.quad_model(alphas, scalings, \
                                amat, bmat, cmat, s0, 1.)

sims = np.column_stack([s1*X1, fx[:, 0]*X1, \
                            -fx[:, 1]*X1, -fx[:, 2]*X1])

# Plot results
plt.close("all")
fig, axs = plt.subplots(nrows=4, figsize=(15, 10), layout="constrained")
for iax, ax in enumerate(axs):
    ax.plot(time, sims[:, iax])

plt.show()
```

# License
The source code and documentation of the pydaisi package is licensed under the
[BSD license](LICENSE.txt).

