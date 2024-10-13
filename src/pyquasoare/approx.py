from itertools import product as prod
import math
import numpy as np

from pyquasoare import has_c_module
if has_c_module():
    import c_pyquasoare
    QUASOARE_EPS = c_pyquasoare.C_QUASOARE_EPS
    QUASOARE_ATOL = c_pyquasoare.C_QUASOARE_ATOL
    QUASOARE_RTOL = c_pyquasoare.C_QUASOARE_RTOL
    QUASOARE_PI = c_pyquasoare.C_QUASOARE_PI
    QUASOARE_NFLUXES_MAX = c_pyquasoare.C_QUASOARE_NFLUXES_MAX
    QUASOARE_ACCURACY = c_pyquasoare.compiler_accuracy_kahan()
else:
    raise ImportError("Cannot run rezeq without C code. Please compile C code.")



def isequal(f1, f2, atol=QUASOARE_ATOL, \
                    rtol=QUASOARE_RTOL):
    """ Check if two values are equal given absolute and relative tolerance. """
    errmax = atol+rtol*np.abs(f1)
    return np.abs(f1-f2)<errmax


def notequal(f1, f2, atol=QUASOARE_ATOL, \
                    rtol=QUASOARE_RTOL):
    """ Check if two values are not equal given absolute and relative tolerance. """
    return 1-isequal(f1, f2, atol, rtol)


def notnull(x):
    """ Check if value is not null. """
    return 1 if x<0 or x>0 else 0


def isnull(x):
    """ Check if value is null. """
    return 1-notnull(x)


def all_scalar(*args):
    """ Check if all arguments are scalar """
    return all([np.isscalar(x) for x in args])


def get_vectors(*args, dtype=np.float64):
    """ Convert all arguments to vector of same length

    Parameters
    -----------
    *args : tuple
        Arguments being either float or vectors of the same length.
    dtype : np.dtype, default np.float64
        Convert inputs to a particular dtype.

    Returns
    -----------
    list of np.ndarray
        List of numpy vectors of the same length.
    """
    v = [np.atleast_1d(x).astype(dtype) for x in args]
    lengths = [len(x) for x in v]
    nval = max(lengths)
    errmess = f"Expected vectors of length 1 or {nval}."
    assert np.all(np.isin(lengths, [1, nval])), errmess
    ones = np.ones(nval)
    return [x[0]*ones if len(x)==1 else x for x in v]


def quad_fun(a, b, c, s):
    """ Quadratic approximation function f(s)=a.s^2+b.s+c

    Parameters
    -----------
    a, b, c, s : float or np.ndarray

    Returns
    -----------
    float or np.ndarray
        Function evaluation.
    """
    if all_scalar(a, b, c, s):
        return c_pyquasoare.quad_fun(a, b, c, s)
    else:
        a, b, c, s, o = get_vectors(a, b, c, s, np.nan)
        ierr = c_pyquasoare.quad_fun_vect(a, b, c, s, o)
        return o


def quad_grad(a, b, c, s):
    """ Gradient of quadratic approximation function df/ds(s)=2a.s+b

    Parameters
    -----------
    a, b, c, s : float or np.ndarray

    Returns
    -----------
    float or np.ndarray
        Gradient function evaluation.
    """
    if all_scalar(a, b, c, s):
        return c_pyquasoare.quad_grad(a, b, c, s)
    else:
        a, b, c, s, o = get_vectors(a, b, c, s, np.nan)
        ierr = c_pyquasoare.quad_grad_vect(a, b, c, s, o)
        return o


def quad_coefficients(alphaj, alphajp1, f0, f1, fm, \
                             approx_opt=1):
    """ Compute the interpolation coefficients for a function over
    the interval [alphaj, alpjajp1].

    The coefficients are obtained by matching the function with the
    quadratic approximation function at three points:
    x = alphaj,
    x = alphajp1,
    x = (alphaj+alphajp1)/2.

    Parameters
    -----------
    alphaj : float
        Lower bound of approximation interval.
    alphajp1 : float
        Upper bound of approximation interval.
    f0 : float
        Function value at x=alphaj.
    f1 : float
        Function value at x=alphajp1.
    fm : float
        Function value at x=(alphaj+alphajp1)/2.
    approx_opt : int, default 1
        Options to restrict the quadratic function fitting:
        0 = linear (i.e. no quadratic term, i.e. ignore fm)
        1 = monotonic (i.e. no zero of the approximation functions)
        2 = free (no restriction)

    Returns
    -----------
    coefs : np.ndarray
        Vector containing the 3 coefficients (a, b, c).

    See Also
    --------
    quad_fun : Quadratic approximation function.

    Examples
    --------
    >>> from pyquasoare import approx
    >>> fun = lambda x: 1-x**6/2
    >>> a0, a1 = 0.6, 1
    >>> f0, f1, fm = fun(a0), fun(a1), fun((a0+a1)/2)
    >>> approx.quad_coefficients(a0, a1, f0, f1, fm, approx_opt=0)
    array([ 0.     , -1.19168,  1.69168])
    >>> approx.quad_coefficients(a0, a1, f0, f1, fm, approx_opt=1)
    array([-2.9792 ,  3.57504, -0.09584])
    >>> approx.quad_coefficients(a0, a1, f0, f1, fm, approx_opt=2)
    array([-3.2648,  4.032 , -0.2672])
    """
    coefs = np.zeros(3)
    ierr = c_pyquasoare.quad_coefficients(approx_opt, alphaj, alphajp1, \
                                            f0, f1, fm, coefs)
    if ierr>0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.quad_coefficients returns {ierr} ({mess})")

    return coefs


def quad_coefficient_matrix(funs, alphas, approx_opt=1):
    """ Compute interpolation coefficients for a set of flux functions and
    multiple interpolation bands.

    Parameters
    -----------
    funs : list of function
        Flux functions.
    alphas : np.ndarray
        Approximation nodes. Vector of length M.
        Should be strictly increasing, i.e. alphas[i]>alphas[i-1]
    approx_opt : int, default 1
        Options to restrict the quadratic function fitting:
        0 = linear (i.e. no quadratic term)
        1 = monotonic (i.e. no zero of the approximation functions)
        2 = free (no restriction)

    Returns
    -------
    a_matrix, b_matrix, c_matrix : np.ndarray
        Approximation coefficient matrices of size [M-1 x N]
        with M=len(alphas) and N=len(funs).

    constants : np.ndarray
        Matrix of constants Delta, tmax:
        Delta = determinant of approximation function (i.e. b^2-4a.c)
        tmax = Validity domain where solution of approximate reservoir
               equation remains valid.

    See Also
    --------
    quad_fun : Quadratic approximation function.

    Examples
    --------
    >>> import numpy as np
    >>> from pyquasoare import approx
    >>> fun = lambda x: 1-x**6/2
    >>> alphas = np.array([0.6, 0.8, 1.])
    >>> amat, bmat, cmat, cst = approx.quad_coefficient_matrix([fun], alphas, approx_opt=1)
    >>> amat
    array([[-1.83755],
           [-4.98155]])
    >>> bmat
    array([[2.03385],
           [7.12215]])
    >>> cmat
    array([[ 0.41788],
           [-1.6406 ]])
    """
    nalphas = len(alphas)
    nfluxes = len(funs)
    if nfluxes>QUASOARE_NFLUXES_MAX:
        raise ValueError(f"Expected nfluxes<{QUASOARE_NFLUXES_MAX}, "\
                            +f"got {nfluxes}.")

    # we add one row at the top end bottom for continuity extension
    a_matrix = np.zeros((nalphas-1, nfluxes))
    b_matrix = np.zeros((nalphas-1, nfluxes))
    c_matrix = np.zeros((nalphas-1, nfluxes))
    constants = np.zeros((nalphas-1, nfluxes, 2))

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

            # Discriminant
            cst = np.zeros(3)
            ierr = c_pyquasoare.quad_constants(a, b, c, cst)
            Delta, qD, sbar = cst

            # Tmax
            t1 = c_pyquasoare.quad_delta_t_max(a, b, c, Delta, qD, sbar, alphaj)
            t2 = c_pyquasoare.quad_delta_t_max(a, b, c, Delta, qD, sbar, alphajp1)
            tmax = min(t1, t2)
            constants[j, ifun, :] = [Delta, tmax]

    return a_matrix, b_matrix, c_matrix, constants


def quad_fun_from_matrix(alphas, a_matrix, b_matrix, c_matrix, x):
    """ Evaluate piecewise quadratic approximation function givien
    interpolation nodes (alphas), interpolation coefficients
    (a, b, c matrices) and evaluation points (x).

    Parameters
    -----------
    alphas : np.ndarray
        Approximation nodes. Vector of length M.
        Should be strictly increasing, i.e. alphas[i]>alphas[i-1]
    a_matrix, b_matrix, c_matrix : np.ndarray
        Interpolation coefficients matrices of size [.
    x : np.ndarray
        Evaluation points of length nval.

    Returns
    -------
    outputs : np.ndarray
        Evaluatiopn of piecewise approximation function
        (size [nval x N]

    See Also
    --------
    quad_fun : Approximation function.
    quad_coefficient_matrix : Obtain interpolation coefficients.

    Examples
    --------
    >>> import numpy as np
    >>> from pyquasoare import approx
    >>> fun = lambda x: 1-x**6/2
    >>> alphas = np.array([0.6, 0.8, 1.])
    >>> amat, bmat, cmat, cst = approx.quad_coefficient_matrix([fun], alphas, approx_opt=1)
    >>> x = np.linspace(alphas[0], alphas[1], 10)
    >>> quad_fun_from_matrix(alphas, amat, bmat, cmat, x)
    array([[0.976672  ],
           [0.9719599 ],
           [0.96543294],
           [0.95709111],
           [0.94693442],
           [0.93496286],
           [0.92117644],
           [0.90557516],
           [0.88815901],
           [0.868928  ]])
    """
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

