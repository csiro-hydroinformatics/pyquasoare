import numpy as np

from pyquasoare import has_c_module, approx
from pyquasoare.approx import QUASOARE_NFLUXES_MAX

if has_c_module():
    import c_pyquasoare
else:
    raise ImportError("Cannot run quasoare without C code. Please compile C code.")


def find_alpha(alphas, u0):
    return c_pyquasoare.find_alpha(alphas, u0)


def eta_fun(x, Delta):
    return c_pyquasoare.eta_fun(x, Delta)


def quad_constants(a, b, c):
    constants = np.zeros(3)
    ierr = c_pyquasoare.quad_constants(a, b, c, constants)
    if ierr>0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.constants returns {ierr} ({mess})")
    return constants


def quad_forward(a, b, c, Delta, qD, sbar, t0, s0, t):
    if np.isscalar(t):
        return c_pyquasoare.quad_forward(a, b, c, Delta, qD, sbar, t0, s0, t)
    else:
        s = np.nan*np.ones_like(t)
        ierr = c_pyquasoare.quad_forward_vect(a, b, c, Delta, qD, sbar, t0, s0, t, s)
        if ierr>0:
            raise ValueError(f"c_pyquasoare.quad_forward_vect returns {ierr}")
        return s


def quad_delta_t_max(a, b, c, Delta, qD, sbar, s0):
    return c_pyquasoare.quad_delta_t_max(a, b, c, Delta, qD, sbar, s0)


def quad_inverse(a, b, c, Delta, qD, sbar, s0, s1):
    if np.isscalar(s1):
        return c_pyquasoare.quad_inverse(a, b, c, Delta, qD, sbar, s0, s1)
    else:
        t = np.nan*np.ones_like(s1)
        ierr = c_pyquasoare.quad_inverse_vect(a, b, c, Delta, qD, sbar, s0, s1, t)
        if ierr>0:
            raise ValueError(f"c_pyquasoare.quad_inverse_vect returns {ierr}")
        return t


def quad_fluxes(a_vector, b_vector, c_vector, \
                        aoj, boj, coj, \
                        Delta, qD, sbar, \
                        t0, t1, s0, s1, fluxes):
    ierr = c_pyquasoare.quad_fluxes(a_vector, b_vector, c_vector, \
                            aoj, boj, coj, Delta, qD, sbar, \
                            t0, t1, s0, s1, fluxes)
    if ierr>0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.quad_fluxes returns {ierr} ({mess})")


def quad_integrate(alphas, scalings, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling, \
                t0, s0, timestep):
    # Initialise
    fluxes = np.zeros(a_matrix_noscaling.shape[1], dtype=np.float64)
    niter = np.zeros(1, dtype=np.int32)
    s1 = np.zeros(1, dtype=np.float64)

    # run
    ierr = c_pyquasoare.quad_integrate(alphas, scalings, \
                    a_matrix_noscaling, b_matrix_noscaling, \
                    c_matrix_noscaling, t0, s0, timestep, niter, s1, fluxes)
    if ierr>0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.quad_integrate returns {ierr} ({mess})")

    return niter[0], s1[0], fluxes


