import numpy as np

from pyrezeq import has_c_module
if has_c_module():
    import c_pyrezeq
else:
    raise ImportError("Cannot run rezeq without C module. Please compile C code")


def model(delta, alphas, scalings, nu, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling, s0):
    fluxes = np.zeros(scalings.shape, dtype=np.float64)
    niter = np.zeros(scalings.shape[0], dtype=np.int32)
    s1 = np.zeros(scalings.shape[0], dtype=np.float64)

    ierr = c_pyrezeq.model(delta, alphas, scalings, \
                    nu_vector, \
                    a_matrix_noscaling, \
                    b_matrix_noscaling, \
                    c_matrix_noscaling, \
                    s0, niter, s1, fluxes)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.run returns {ierr}")

    return niter, s1, fluxes


