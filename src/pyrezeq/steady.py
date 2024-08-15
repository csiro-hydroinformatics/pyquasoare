import math
import numpy as np

from pyrezeq import has_c_module, approx

if has_c_module():
    import c_pyrezeq
    REZEQ_EPS = c_pyrezeq.get_eps()
    REZEQ_NFLUXES_MAX = c_pyrezeq.get_nfluxes_max()
else:
    raise ImportError("Cannot run rezeq without C code. Please compile C code.")


def quad_steady(a, b, c):
    if approx.all_scalar(a, b, c):
        steady = np.zeros(2)
        ierr = c_pyrezeq.quad_steady(a, b, c, steady)
    else:
        a, b, c = approx.get_vectors(a, b, c)
        steady = np.zeros((len(a), 2))
        ierr = c_pyrezeq.quad_steady_vect(a, b, c, steady)
    return steady


def quad_steady_scalings(alphas, scalings, \
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
        a = scalings@a_matrix_noscaling[j]
        b = scalings@b_matrix_noscaling[j]
        c = scalings@c_matrix_noscaling[j]
        stdy = quad_steady(a, b, c)

        # Keep steady solution within band
        a0, a1 = alphas[[j, j+1]]
        out_of_range = (stdy-a0)*(a1-stdy)<0
        stdy[out_of_range] = np.nan
        steady[:, 2*j:(2*j+2)] = stdy

    # Reorder and remove nan columns
    steady = np.sort(steady, axis=1)

    hasvalid = np.any(~np.isnan(steady), axis=0)
    steady = steady[:, hasvalid]

    return steady


