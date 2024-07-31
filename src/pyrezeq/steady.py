import math
import numpy as np

from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit, logit, softmax
from scipy.integrate import solve_ivp

from pyrezeq import has_c_module, approx

if has_c_module():
    import c_pyrezeq
    REZEQ_EPS = c_pyrezeq.get_eps()
    REZEQ_NFLUXES_MAX = c_pyrezeq.get_nfluxes_max()
else:
    raise ImportError("Cannot run rezeq without C code. Please compile C code.")


def steady_state(nu, a, b, c):
    if approx.all_scalar(a, b, c):
        steady = np.zeros(2)
        ierr = c_pyrezeq.steady_state(nu, a, b, c, steady)
    else:
        a, b, c = approx.get_vectors(a, b, c)
        steady = np.zeros((len(a), 2))
        ierr = c_pyrezeq.steady_state_vect(nu, a, b, c, steady)
    return steady


def steady_state_scalings(alphas, nu, scalings, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling):
    """ Compute steady states using scalings """
    # Check inputs
    nalphas = len(alphas)
    assert nalphas>2
    nval, nfluxes = scalings.shape
    for m in [a_matrix_noscaling, b_matrix_noscaling, c_matrix_noscaling]:
        assert len(m) == nalphas-1
        assert m.shape[1] == nfluxes

    # Potentially a max of 2 x (nalphas-1) solution if
    # there are 2 solutions for each band (very unlikely)
    steady = np.zeros((nval, 2*(nalphas-1)))

    for j in range(0, nalphas-1):
        a0 = scalings@a_matrix_noscaling[j]
        b0 = scalings@b_matrix_noscaling[j]
        c0 = scalings@c_matrix_noscaling[j]
        stdy = steady_state(nu, a0, b0, c0)

        # Keep steady solution within band
        a0, a1 = alphas[[j, j+1]]
        out_of_range = (stdy-a0)*(a1-stdy)<0
        stdy[out_of_range] = np.nan
        steady[:, 2*j:(2*j+2)] = stdy

    # Set up bands
    b = np.repeat(np.arange(nalphas-1), 2)
    bands = np.repeat(b[None, :], nval, axis=0)

    # Sort values and eliminate columns with nan only
    isort = np.argsort(steady, axis=1)
    idx = np.arange(nval)[:, None]
    steady, bands = steady[idx, isort], bands[idx, isort]

    has_valid = (np.isnan(steady)).sum(axis=0)==0
    steady, bands = steady[:, has_valid], bands[:, has_valid]

    return steady, bands


