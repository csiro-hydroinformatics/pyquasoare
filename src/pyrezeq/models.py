import warnings
import numpy as np

from pyrezeq import has_c_module
if has_c_module():
    import c_pyrezeq
else:
    raise ImportError("Cannot run rezeq without C module. Please compile C code")

ERRORS = ["ignore", "raise", "warn"]


def quad_model(alphas, scalings, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling, s0, timestep, \
                errors="ignore"):
    assert errors in ERRORS
    nval = scalings.shape[0]
    fluxes = np.zeros(scalings.shape, dtype=np.float64)
    niter = np.zeros(nval, dtype=np.int32)
    s1 = np.zeros(nval, dtype=np.float64)

    ierr = c_pyrezeq.quad_model(alphas, scalings, \
                    a_matrix_noscaling, \
                    b_matrix_noscaling, \
                    c_matrix_noscaling, \
                    s0, timestep, niter, s1, fluxes)

    if ierr>0:
        mess = c_pyrezeq.get_error_message(ierr).decode()
        errmess = f"c_pyrezeq.quad_model returns {ierr} ({mess})"

        if errors=="raise":
            raise ValueError(errmess)

        elif errors=="warn":
            warnings.warn(errmess)

    return niter, s1, fluxes


