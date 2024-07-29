import math
import numpy as np

from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit, logit, softmax
from scipy.integrate import solve_ivp

from pyrezeq import has_c_module
if has_c_module():
    import c_pyrezeq
    REZEQ_EPS = c_pyrezeq.get_eps()
    REZEQ_NFLUXES_MAX = c_pyrezeq.get_nfluxes_max()
else:
    REZEQ_EPS = 1e-10
    REZEQ_NFLUXES_MAX = 20

# --- approximation functions ----

def approx_fun(nu, a, b, c, s):
    if isinstance(s, np.ndarray):
        ds = np.zeros_like(s)
        ierr = c_pyrezeq.approx_fun_vect(nu, a, b, c, s, ds)
        return ds
    else:
        return c_pyrezeq.approx_fun(nu, a, b, c, s)


def approx_jac(nu, a, b, c, s):
    if isinstance(s, np.ndarray):
        ds = np.zeros_like(s)
        ierr = c_pyrezeq.approx_jac_vect(nu, a, b, c, s, ds)
        return ds
    else:
        return c_pyrezeq.approx_jac(nu, a, b, c, s)


def steady_state(nu, a, b, c):
    if isinstance(a, np.ndarray):
        n = len(a)
        steady = np.zeros((n, 2))
        ierr = c_pyrezeq.steady_state_vect(nu, a, b, c, steady)
    else:
        steady = np.zeros(2)
        ierr = c_pyrezeq.steady_state(nu, a, b, c, steady)
    return steady


def steady_state_scalings(alphas, nu, scalings, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling):
    """ Compute steady states using scalings """
    # Check inputs
    nalphas = len(alphas)
    nval, nfluxes = scalings.shape
    for m in [a_matrix_noscaling, b_matrix_noscaling, c_matrix_noscaling]:
        assert len(m) == nalphas-1
        assert m.shape[1] == nfluxes

    steady = np.nan*np.zeros((nval, 2))
    for j in range(1, nalphas-1):
        a0 = scalings@a_matrix_noscaling[j]
        b0 = scalings@b_matrix_noscaling[j]
        c0 = scalings@c_matrix_noscaling[j]
        nu = np.ones_like(a0)*nu
        s = steady_state(nu, a0, b0, c0)

        # eliminates steady states outside bounds
        s[s<alphas[j]] = np.nan
        s[s>alphas[j+1]] = np.nan
        isok = ~np.isnan(s)
        steady[isok] = s[isok]

    return np.array(steady)


def get_coefficients(fun, alphaj, alphajp1, nu, epsilon):
    """ Find approx coefficients for the interval [alphaj, alpjajp1]
        nu is the non-linearity coefficient.
        epsilon is the option controlling the location of the third constraint
    """
    assert alphajp1>alphaj
    assert nu>0
    assert epsilon>0 and epsilon<1

    fa = lambda a, b, c, x: approx_fun(nu, a, b, c, x)
    am = (1-epsilon)*alphaj+epsilon*alphajp1
    aa = [alphaj, am, alphajp1]

    X  = [[fa(1, 0, 0, a), fa(0, 1, 0, a), fa(0, 0, 1, a)] for a in aa]
    y = [fun(a) for a in aa]

    return np.linalg.solve(X, y)


def get_coefficients_matrix(funs, alphas, nu=1, epsilon=0.5):
    """ Generate coefficient matrices for flux functions """
    nalphas = len(alphas)
    nfluxes = len(funs)
    if nfluxes>REZEQ_NFLUXES_MAX:
        raise ValueError(f"Expected nfluxes<{REZEQ_NFLUXES_MAX}, "\
                            +f"got {nfluxes}.")

    # we add one row at the top end bottom for continuity extension
    a_matrix = np.zeros((nalphas-1, nfluxes))
    b_matrix = np.zeros((nalphas-1, nfluxes))
    c_matrix = np.zeros((nalphas-1, nfluxes))

    for j in range(nalphas-1):
        alphaj, alphajp1 = alphas[[j, j+1]]
        for ifun, f in enumerate(funs):
            a, b, c = get_coefficients(f, alphaj, alphajp1, nu, epsilon)
            a_matrix[j, ifun] = a
            b_matrix[j, ifun] = b
            c_matrix[j, ifun] = c

    return a_matrix, b_matrix, c_matrix



def check_coefficient_matrix(funs, alphas, nu, a_matrix, b_matrix, c_matrix,\
                                                                ninterp=10000):
    """ Check no spurious steady state is present """
    nalphas = len(alphas)
    nfluxes = len(funs)
    has_problem = np.zeros((nalphas-1, nfluxes))
    for j in range(nalphas-1):
        alphaj, alphajp1 = alphas[[j, j+1]]
        xx = np.linspace(alphaj, alphajp1, ninterp)

        for ifun, f in enumerate(funs):
            yy = f(xx)
            a = a_matrix[j, ifun]
            b = b_matrix[j, ifun]
            c = c_matrix[j, ifun]
            yya = approx_fun(nu, a, b, c, xx)

            notzero = np.abs(yya)>REZEQ_EPS
            has_problem[j, ifun] = int(np.any((yy*yya<=-REZEQ_EPS) & notzero))

    return has_problem


def approx_fun_from_matrix(alphas, nu, a_matrix, b_matrix, c_matrix, s):
    nalphas = len(alphas)
    nfluxes = a_matrix.shape[1]
    assert a_matrix.shape[0] == nalphas-1
    assert b_matrix.shape == a_matrix.shape
    assert c_matrix.shape == a_matrix.shape

    s = np.atleast_1d(s)
    outputs = np.nan*np.zeros((len(s), a_matrix.shape[1]))

    # Outside of alpha bounds
    idx = s<alphas[0]
    if idx.sum()>0:
        o = [approx_fun(nu, a, b, c, alphas[0]) for a, b, c \
                    in zip(a_matrix[0], b_matrix[0], c_matrix[0])]
        outputs[idx] = np.column_stack(o)

    idx = s>alphas[-1]
    if idx.sum()>0:
        o = [approx_fun(nu, a, b, c, alphas[-1]) for a, b, c \
                    in zip(a_matrix[-1], b_matrix[-1], c_matrix[-1])]
        outputs[idx] = np.column_stack(o)

    # Inside alpha bounds
    for j in range(nalphas-1):
        alphaj, alphajp1 = alphas[[j, j+1]]
        idx = (s>=alphaj-1e-10)&(s<=alphajp1+1e-10)
        if idx.sum()==0:
            continue

        for i in range(nfluxes):
            a = a_matrix[j, i]
            b = b_matrix[j, i]
            c = c_matrix[j, i]
            outputs[idx, i] = approx_fun(nu, a, b, c, s[idx])

    return outputs


## -- Integration functions --

def integrate_forward(nu, a, b, c, t0, s0, t):
    if isinstance(t, np.ndarray):
        s = np.zeros_like(t)
        ierr = c_pyrezeq.integrate_forward_vect(nu, a, b, c, t0, s0, t, s)
        return s
    else:
        return c_pyrezeq.integrate_forward(nu, a, b, c, t0, s0, t)


def integrate_delta_t_max(nu, a, b, c, s0):
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



def run(delta, alphas, scalings, \
                nu_vector, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling, \
                s0):
    fluxes = np.zeros(scalings.shape, dtype=np.float64)
    niter = np.zeros(scalings.shape[0], dtype=np.int32)
    s1 = np.zeros(scalings.shape[0], dtype=np.float64)
    ierr = c_pyrezeq.run(delta, alphas, scalings, \
                    nu_vector, \
                    a_matrix_noscaling, \
                    b_matrix_noscaling, \
                    c_matrix_noscaling, \
                    s0, niter, s1, fluxes)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.run returns {ierr}")

    return niter, s1, fluxes


def quadrouting(delta, theta, q0, s0, inflows, \
                    engine="C"):
    inflows = np.array(inflows).astype(np.float64)
    outflows = np.zeros_like(inflows)
    ierr = c_pyrezeq.quadrouting(delta, theta, q0, s0, inflows, outflows)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.quadrouting returns {ierr}")

    return outflows



