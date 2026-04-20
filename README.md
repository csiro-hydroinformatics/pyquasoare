![QuaSoaRe](quasoare_icon.png)

# pyquasoare
 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13928253.svg)](https://doi.org/10.5281/zenodo.13928253) 
 [![Build pyquasoare](https://github.com/csiro-hydroinformatics/pyquasoare/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/csiro-hydroinformatics/pyquasoare/actions/workflows/python-package-conda.yml) 
![Coverage](https://gist.githubusercontent.com/jlerat/cfffdda1ba3456d850a0d75eca8dab89/raw/coverage_badge.svg)

Python and C package to solve the reservoir differential equation using a
piecewise quadratic interpolation following the QuaSoARe method.

# What is pyquasoare?
This package implements the Quadratic Solution of the Approximate Reservoir 
Equation (QuaSoARe) method described in the following paper:
Lerat, J. (2025), Technical note: Quadratic Solution of the Approximate Reservoir Equation (QuaSoARe), Hydrol. Earth Syst. Sci., 29, 2003–2021, https://doi.org/10.5194/hess-29-2003-2025, 2025.

# Installation
- Create a suitable python environment. We recommend using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) combined with the environment specification provided in the [env\_pyquasoare.yml](env_pyquasoare.yml) file in this repository.
- Git clone this repository and run `pip install .`

# Basic use

## Approximation of a function with a piecewise quadratic function:
```python
import numpy as np
from pyquasoare import approx

# We want to approximate a 6th order polynomial+
funs = [lambda x: x - x**2 + x**6]

# We select 20 nodes over [0, 1]
nalphas = 20
alphas = np.linspace(0., 1., nalphas)
coefs = approx.quad_coefficient_matrix(funs, alphas)

# Test the approximation
xx = np.linspace(0, 1, 200)
yy = approx.quad_fun_from_matrix(alphas, coefs, xx)
```

## Simulation using the production store of the [GR4J](https://www.sciencedirect.com/science/article/pii/S0022169403002257) daily rainfall-runoff model using QuaSoAre

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyquasoare import approx, models

# The production store of the GR4J model is characterised by
# the following differential equation:
# dS / dt = P (1 - [S/X1]**2) - E S/X1 (2 - S/X1) - a (S/X1)**5
# where S is the store volume (mm), X1 is the store capacity,
# P and E are the rainfall and evapotranspiration (mm/day) and
# a is a constant set to 1/2.25**4/4 (~9.75e-3).
#
# If we introduce the following variables:
# p = P / X1
# e = E / X1
# u = S / X1
# the previous equation becomes:
# du / dt = p (1 - u**2) - e u (2 - u) - a u**5
# this equation has 3 fluxes:
# * rainfall infiltrated into the store : p (1 - u**2)
# * actual evapotranspiration : - e u (2 - u)
# * percolation : -a u**5

X1 = 400

fluxes = [
    lambda u: 1 - u**2,
    lambda u: -u * (2 - u),
    lambda u: -(1. / 2.25)**5 / 4 * u**5
]

# We are now solving this differential equation with QuaSoARe:

# 1. Definition of interpolation points
nalphas = 20
alphas = np.linspace(0., 1.2, nalphas)

# 2. Quadratic piecewise interpolation of the flux functions
coefs = approx.quad_coefficient_matrix(fluxes, alphas)

# 3. Processing rainfall and PET data
# .. creating random data for the example
nval = 1000
time = pd.date_range("2000-01-01", freq="D", periods=nval)
rain = np.maximum(np.random.exponential(8, size=nval) - 10, 0)
evap = 2 + 1.5 * (np.sin(np.arange(nval) / 365.25 * 2 * 6.28) + 1)/2

# GR4J applies an interception function. This leads to
rain_intercept = np.maximum(rain - evap, 0.)
evap_intercept = np.maximum(evap - rain, 0.)

# The scalings indicated below correspond to variables
# 'p' and 'e' in the equations above:
scalings = np.column_stack([rain_intercept / X1,
                            evap_intercept / X1,
                            np.ones(nval)])

# Run the model using QuaSoare
s0 = 1./2
niter, s1, fx = models.quad_model(alphas, scalings,
                                  coefs, s0, 1.)

# All fluxes computed by QuaSoARe needs to be rescaled
# by X1 because the equation was solved for variables
# divided by X1 (see equations above)
sims = pd.DataFrame(np.column_stack([s1 * X1,
                                    fx[:, 0] * X1,
                                    -fx[:, 1] * X1,
                                    -fx[:, 2] * X1]),
                    index=time,
                    columns=["store",
                             "infiltrated rain",
                             "actual ET",
                             "percolation"])
# Effective rainfall is the remaining of rainfall minus
# infiltrated rainfall
sims.loc[:, "effective rain"] = rain - sims.iloc[:, 1]

# Plot results for the first 100 days
plt.close("all")
fig = plt.figure(figsize=(10, 10), layout="constrained")
axs = fig.subplot_mosaic([[vn] for vn in sims.columns],
                         sharex=True)
for varname, ax in axs.items():
    sims.loc[:, varname].plot(ax=ax)
    ax.set(title=varname)

fig.savefig("simulation.png")
```

# Generation of results supporting the QuaSoARe paper
All results presented in the QuaSoARe paper can be generated by running the
python script [models\_run.py](scripts/quasoare_paper_2024/models_run.py). This
script applies QuaSoARe to a set of test cases defined by the script argument '-t'
which varies from 0 to 29. The test cases includes application of QuaSoARe to
* 6 catchments locaed in Eastern Australia,
* 5 hydrological models

To run all cases, the script needs to be launched within a loop as follows
(assuming Linux/Mac OS bash script):
```bash
for taskid in {0..29}; do
    python scripts/quasoare_paper_2024/models_run.py -t taskid
done    
```
Once the results are generated, the figures of the paper can be generated using
the script [figures\_generate\_all.py](scripts/quasoare_paper_2024/figures_generate_all.py).

# Attribution
This project is licensed under the [MIT License](LICENSE), which allows for free use, modification, and distribution of the code under the terms of the license.

For proper citation of this project, please refer to the [CITATION.cff](CITATION.cff) file, which provides guidance on 
how to cite the software and relevant publications.

