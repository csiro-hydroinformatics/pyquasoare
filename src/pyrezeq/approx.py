from itertools import product as prod
import math
import numpy as np

from pyrezeq import has_c_module
if has_c_module():
    import c_pyrezeq
    REZEQ_EPS = c_pyrezeq.C_REZEQ_EPS
    REZEQ_ATOL = c_pyrezeq.C_REZEQ_ATOL
    REZEQ_RTOL = c_pyrezeq.C_REZEQ_RTOL
    REZEQ_PI = c_pyrezeq.C_REZEQ_PI
    REZEQ_NFLUXES_MAX = c_pyrezeq.C_REZEQ_NFLUXES_MAX
    REZEQ_ACCURACY = c_pyrezeq.compiler_accuracy_kahan()
else:
    raise ImportError("Cannot run rezeq without C code. Please compile C code.")



def isequal(f1, f2, atol=REZEQ_ATOL, \
                    rtol=REZEQ_RTOL):
    """ Checking if two values are equal """
    errmax = atol+rtol*np.abs(f1)
    return np.abs(f1-f2)<errmax


def notequal(f1, f2, atol=REZEQ_ATOL, \
                    rtol=REZEQ_RTOL):
    return 1-isequal(f1, f2, atol, rtol)


def notnull(x):
    return 1 if x<0 or x>0 else 0

def isnull(x):
    return 1-notnull(x)

def all_scalar(*args):
    """ Check if all arguments are scalar """
    return all([np.isscalar(x) for x in args])


def get_vectors(*args, dtype=np.float64):
    """ Convert all arguments to vector of same length """
    v = [np.atleast_1d(x).astype(dtype) for x in args]
    nval = max([len(x) for x in v])
    ones = np.ones(nval)
    return [x[0]*ones if len(x)==1 else x for x in v]


def quad_fun(a, b, c, s):
    """ Approximation function f=a.s^2+b.s+c """
    if all_scalar(a, b, c, s):
        return c_pyrezeq.quad_fun(a, b, c, s)
    else:
        a, b, c, s, o = get_vectors(a, b, c, s, np.nan)
        ierr = c_pyrezeq.quad_fun_vect(a, b, c, s, o)
        return o


def quad_grad(a, b, c, s):
    """ Gradient of approximation function f=2a.s+b """
    if all_scalar(a, b, c, s):
        return c_pyrezeq.quad_grad(a, b, c, s)
    else:
        a, b, c, s, o = get_vectors(a, b, c, s, np.nan)
        ierr = c_pyrezeq.quad_grad_vect(a, b, c, s, o)
        return o


def quad_coefficients(alphaj, alphajp1, f0, f1, fm, approx_opt=1):
    """ Find approx coefficients for the interval [alphaj, alpjajp1]
        Fits the approx fun at x=a0, x=a1 and x=(a0+a1)/2.
        approx_opt:
            0 = linear interpolation
            1 = monotonous quadratic interpolation
            2 = quadratic interpolation with no constraint
    """
    coefs = np.zeros(3)
    ierr = c_pyrezeq.quad_coefficients(approx_opt, alphaj, alphajp1, f0, f1, fm, coefs)
    return coefs


def quad_coefficient_matrix(funs, alphas, approx_opt=1):
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
            f0 = f(alphaj)
            f1 = f(alphajp1)
            fm = f((alphaj+alphajp1)/2)
            a, b, c = quad_coefficients(alphaj, alphajp1, f0, f1, fm, \
                                        approx_opt)

            a_matrix[j, ifun] = a
            b_matrix[j, ifun] = b
            c_matrix[j, ifun] = c

    return a_matrix, b_matrix, c_matrix


def quad_fun_from_matrix(alphas, a_matrix, b_matrix, c_matrix, x):
    nalphas = len(alphas)
    nfluxes = a_matrix.shape[1]
    assert np.all(np.diff(alphas)>0)
    assert a_matrix.shape[0] == nalphas-1
    assert b_matrix.shape == a_matrix.shape
    assert c_matrix.shape == a_matrix.shape

    x = np.atleast_1d(x)
    outputs = np.nan*np.zeros((len(x), a_matrix.shape[1]))

    # Outside of alpha bounds
    alpha_min, alpha_max = alphas[0], alphas[-1]
    idx_low = x < alpha_min
    idx_high = x > alpha_max

    for i in range(nfluxes):
        if idx_low.sum()>0:
            # Linear trend in low extrapolation
            al, bl, cl = a_matrix[0, i], b_matrix[0, i], c_matrix[0, i]
            g = quad_grad(al, bl, cl, alpha_min)
            cl = quad_fun(al, bl, cl, alpha_min)-g*alpha_min
            bl = g
            al = 0
            o = quad_fun(al, bl, cl, x[idx_low])
            outputs[idx_low, i] = o

        if idx_high.sum()>0:
            # Linear trend in high extrapolation
            ah, bh, ch = a_matrix[-1, i], b_matrix[-1, i], c_matrix[-1, i]
            g = quad_grad(ah, bh, ch, alpha_max)
            ch = quad_fun(ah, bh, ch, alpha_max)-g*alpha_max
            bh = g
            ah = 0
            o = quad_fun(ah, bh, ch, x[idx_high])
            outputs[idx_high, i] = o

    # Inside alpha bounds
    for j in range(nalphas-1):
        alphaj, alphajp1 = alphas[[j, j+1]]
        idx = (x>=alphaj-1e-10) & (x<=alphajp1+1e-10)
        if idx.sum()==0:
            continue

        for i in range(nfluxes):
            a = a_matrix[j, i]
            b = b_matrix[j, i]
            c = c_matrix[j, i]
            outputs[idx, i] = quad_fun(a, b, c, x[idx])

    return outputs

