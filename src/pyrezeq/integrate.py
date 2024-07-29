import numpy as np

from pyrezeq import has_c_module
if has_c_module():
    import c_pyrezeq
else:
    raise ImportError("Cannot run rezeq without C code. Please compile C code.")


def integrate_forward(nu, a, b, c, t0, s0, t):
    if isinstance(t, np.ndarray):
        s = np.zeros_like(t)
        ierr = c_pyrezeq.integrate_forward_vect(nu, a, b, c, t0, s0, t, s)
        return s
    else:
        return c_pyrezeq.integrate_forward(nu, a, b, c, t0, s0, t)


def delta_t_max(nu, a, b, c, s0):
    return c_pyrezeq.integrate_delta_t_max(nu, a, b, c, s0)


def integrate_inverse(nu, a, b, c, s0, s1):
    if isinstance(s1, np.ndarray):
        t = np.zeros_like(s1)
        ierr = c_pyrezeq.integrate_inverse_vect(nu, a, b, c, s0, s1, t)
        return t
    else:
        return c_pyrezeq.integrate_inverse(nu, a, b, c, s0, s1)


def find_alpha(alphas, u0):
    return c_pyrezeq.find_alpha(alphas, u0)



def increment_fluxes(nu, a_vector, b_vector, c_vector, \
                        aoj, boj, coj, \
                        t0, t1, s0, s1, fluxes):

    ierr = c_pyrezeq.increment_fluxes(nu, \
                            a_vector, b_vector, c_vector, \
                            aoj, boj, coj, \
                            t0, t1, s0, s1, fluxes)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.increment_fluxes returns {ierr}")


def integrate(alphas, scalings, nu, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling, \
                t0, s0, delta):
    # Initialise
    fluxes = np.zeros(a_matrix_noscaling.shape[1], dtype=np.float64)
    niter = np.zeros(1, dtype=np.int32)
    s1 = np.zeros(1, dtype=np.float64)
    nus = nu*np.ones(len(alphas)-1)

    # run
    ierr = c_pyrezeq.integrate(alphas, scalings, nus, \
                    a_matrix_noscaling, b_matrix_noscaling, \
                    c_matrix_noscaling, t0, s0, delta, niter, s1, fluxes)
    if ierr>0:
        mess = c_pyrezeq.get_error_message(ierr).decode()
        raise ValueError(f"c_pyrezeq.integrate returns {ierr} ({mess})")

    return niter[0], s1[0], fluxes


