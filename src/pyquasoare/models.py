import warnings
import numpy as np

from pyquasoare import has_c_module
if has_c_module():
    import c_pyquasoare
else:
    raise ImportError("Cannot run quasoare without C module. Please compile C code")

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
    ierrors = np.int32(ERRORS.index(errors))

    ierr = c_pyquasoare.quad_model(ierrors, alphas, scalings, \
                    a_matrix_noscaling, \
                    b_matrix_noscaling, \
                    c_matrix_noscaling, \
                    s0, timestep, niter, s1, fluxes)

    if errors=="raise" and ierr>0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.quad_model returns {ierr} ({mess})")

    if errors=="warn":
        if np.any(niter<0):
            nerr = (niter<0).sum()
            mess = f"{nerr} errors when running c_pyquasoare.quad_model"
            warnings.warn(mess)

    return niter, s1, fluxes


