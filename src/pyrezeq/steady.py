import math
import numpy as np

from pyrezeq import has_c_module, approx, models

if has_c_module():
    import c_pyrezeq
else:
    raise ImportError("Cannot run rezeq without C code. Please compile C code.")


def quad_steady(a, b, c):
    if approx.all_scalar(a, b, c):
        stdy = np.zeros(2)
        ierr = c_pyrezeq.quad_steady(a, b, c, stdy)
    else:
        a, b, c = approx.get_vectors(a, b, c)
        stdy = np.zeros((len(a), 2))
        ierr = c_pyrezeq.quad_steady_vect(a, b, c, stdy)
    return stdy


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

    # Potentially a max of 2 x (nalphas+1) solutions
    # over nalphas-1 bands and 2 extrpolation if
    # there are 2 solutions for each band
    # (only possible for non-monotonous fuynctions)
    steady = np.zeros((nval, 2*(nalphas+1)))

    for j in range(-1, nalphas):
        if j>=0 and j<nalphas-1:
            # General case
            a = scalings@a_matrix_noscaling[j]
            b = scalings@b_matrix_noscaling[j]
            c = scalings@c_matrix_noscaling[j]
            a0, a1 = alphas[[j, j+1]]

        elif j==-1:
            # Low extrapolation - Linear
            a = scalings@a_matrix_noscaling[0]
            b = scalings@b_matrix_noscaling[0]
            c = scalings@c_matrix_noscaling[0]
            alpha_min = alphas[0]
            grad = approx.quad_grad(a, b, c, alpha_min)

            c = approx.quad_fun(a, b, c, alpha_min)-grad*alpha_min
            b = grad
            a = 0.*grad
            a0, a1 = -np.inf, alpha_min

        elif j==nalphas-1:
            # high extrapolation - Linear
            a = scalings@a_matrix_noscaling[-1]
            b = scalings@b_matrix_noscaling[-1]
            c = scalings@c_matrix_noscaling[-1]
            alpha_max = alphas[-1]
            grad = approx.quad_grad(a, b, c, alpha_max);

            c = approx.quad_fun(a, b, c, alpha_max)-grad*alpha_max
            b = grad
            a = 0.*grad
            a0, a1 = alpha_max, np.inf

        stdy = quad_steady(a, b, c)

        # Keep steady solution within band
        # .. ignore nan
        with np.errstate(invalid="ignore"):
            out_of_range = (stdy-a0)*(a1-stdy)<0
        stdy[out_of_range] = np.nan
        jc = j+1
        steady[:, 2*jc:(2*jc+2)] = stdy

    # Reorder and remove nan columns
    steady = np.sort(steady, axis=1)
    hasvalid = np.any(~np.isnan(steady), axis=0)
    hasvalid = [0] if hasvalid.sum()==0 else hasvalid
    steady = steady[:, hasvalid]

    return steady

