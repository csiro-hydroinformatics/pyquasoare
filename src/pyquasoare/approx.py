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
    raise ImportError("Cannot run rezeq without C code."
                      + " Please compile C code.")


def isequal(f1, f2, atol=QUASOARE_ATOL, rtol=QUASOARE_RTOL):
    """ Check if two values are equal given absolute
    and relative tolerance.
    """
    errmax = atol+rtol*np.abs(f1)
    return np.abs(f1-f2) < errmax


def notequal(f1, f2, atol=QUASOARE_ATOL, rtol=QUASOARE_RTOL):
    """ Check if two values are not equal given absolute
    and relative tolerance.
    """
    return 1-isequal(f1, f2, atol, rtol)


def notnull(x):
    """ Check if value is not null. """
    return 1 if x < 0 or x > 0 else 0


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
    return [x[0]*ones if len(x) == 1 else x for x in v]


def quad_fun(coefs, s, out=None):
    """ Quadratic approximation function f(s) = a * s^2 + b * s + c

    Parameters
    -----------
    coefs : numpy.ndarray
        3 coefficients.
    s : numpy.ndarray
        Values where quadratic function is computed.

    Returns
    -----------
    out : np.ndarray
        Quadratic function evaluation.
    """
    s = np.atleast_1d(s)
    out = np.zeros(len(s)) if out is None else out
    c_pyquasoare.quad_fun(coefs, s, out)
    return out


def quad_grad(coefs, s, out=None):
    """ Gradient of quadratic approximation function df/ds(s) = 2 * a * s + b

    Parameters
    -----------
    coefs : numpy.ndarray
        3 coefficients.
    s : numpy.ndarray
        Values where quadratic function gradient is computed.

    Returns
    -----------
    out : np.ndarray
        Quadratic function evaluation.
    """
    s = np.atleast_1d(s)
    out = np.zeros(len(s)) if out is None else out
    c_pyquasoare.quad_grad(coefs, s, out)
    return out


def quad_coefficients(alphas, falphas, fmid, approx_opt=1,
                      out=None):
    """ Compute the interpolation coefficients for a function over
    the interval [alphaj, alpjajp1].

    The coefficients are obtained by matching the function with the
    quadratic approximation function at three points:
    1. alphas[j],
    2. alphas[j + 1]
    3. (alphas[j] + alphas[j + 1]) / 2

    Parameters
    -----------
    alphas : numpy.ndarray
        Interpolation nodes
    falphas : float
        Function values at interpolation nodes
    fmid : numpy.ndarray
        Function value at mid points between interpolation nodes
    approx_opt : int, default 1
        Options to restrict the quadratic function fitting:
        0 = linear (i.e. no quadratic term, i.e. ignore fm)
        1 = monotonic (i.e. no zero of the approximation functions)
        2 = free (no restriction)
    out : numpy.ndarray
        In place output result.
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
    nalphas = len(alphas)
    coefs = np.zeros((nalphas - 1, 3)) if out is None else out
    ierr = c_pyquasoare.quad_coefficients(approx_opt, alphas,
                                          falphas, fmid, coefs)
    if ierr > 0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError("c_pyquasoare.quad_coefficients"
                         + f" returns {ierr} ({mess})")
    return coefs


def quad_coefficient_matrix(funs, alphas, approx_opt=1, out=None):
    """ Compute interpolation coefficients for a set of flux functions and
    multiple interpolation bands.

    Parameters
    -----------
    funs : list of function
        Flux functions (list of N elements).
    alphas : numpy.ndarray
        Approximation nodes. Vector of length M.
        Should be strictly increasing, i.e. alphas[i]>alphas[i-1]
    approx_opt : int, default 1
        Options to restrict the quadratic function fitting:
        0 = linear (i.e. no quadratic term)
        1 = monotonic (i.e. no zero of the approximation functions)
        2 = free (no restriction)
    out : numpy.ndarray
        A location into which the result is stored. Must have a shape
        equal to (N, M-1, 3).
    Returns
    -------
    coefs : np.ndarray
        Approximation coefficient matrix of size [N, M-1, 3].

    See Also
    --------
    quad_fun : Quadratic approximation function.

    Examples
    --------
    >>> import numpy as np
    >>> from pyquasoare import approx
    >>> fun = lambda x: 1-x**6/2
    >>> alphas = np.array([0.6, 0.8, 1.])
    >>> coefs = approx.quad_coefficient_matrix([fun], alphas, approx_opt=1)
    >>> coefs[0, :, 0]
    array([[-1.83755],
           [-4.98155]])
    >>> coefs[0, :, 1]
    array([[2.03385],
           [7.12215]])
    >>> coefs[0, :, 2]
    array([[ 0.41788],
           [-1.6406 ]])
    """
    nalphas = len(alphas)
    nfluxes = len(funs)
    if nfluxes > QUASOARE_NFLUXES_MAX:
        raise ValueError(f"Expected nfluxes<{QUASOARE_NFLUXES_MAX}, "
                         + f"got {nfluxes}.")

    coefs = np.zeros((nfluxes, nalphas - 1, 3)) if out is None else out
    mid = (alphas[1:] + alphas[:-1]) / 2

    for ifun, fun in enumerate(funs):
        falphas = fun(alphas)
        fmid = fun(mid)
        quad_coefficients(alphas, falphas, fmid,
                          approx_opt, out=coefs[ifun])
    return coefs


def quad_fun_from_matrix(alphas, coefs, x, out=None):
    """ Evaluate piecewise quadratic approximation function givien
    interpolation nodes (alphas), interpolation coefficients
    (a, b, c matrices) and evaluation points (x).

    Parameters
    -----------
    alphas : np.ndarray
        Approximation nodes. Vector of length M.
        Should be strictly increasing, i.e. alphas[i]>alphas[i-1]
    coefs : np.ndarray
        Interpolation coefficients matrices of size [N, M - 1, 3].
    x : np.ndarray
        Evaluation points of length P.
    out : numpy.ndarray
        A location into which the result is stored. Must have a shape
        equal to (P, N).

    Returns
    -------
    outputs : np.ndarray
        Evaluatiopn of piecewise approximation function
        (size [P x N]

    See Also
    --------
    quad_fun : Approximation function.
    quad_coefficient_matrix : Obtain interpolation coefficients.

    Examples
    --------
    >>> import numpy as np
    >>> from pyquasoare import approx
    >>> fun = lambda x: 1 - x**6/2
    >>> alphas = np.array([0.6, 0.8, 1.])
    >>> coefs = approx.quad_coefficient_matrix([fun], alphas, approx_opt=1)
    >>> x = np.linspace(alphas[0], alphas[1], 10)
    >>> quad_fun_from_matrix(alphas, coefs, x)
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
    nfluxes = coefs.shape[0]
    assert np.all(np.diff(alphas) > 0)
    assert coefs.shape == (nfluxes, nalphas-1, 3)

    x = np.atleast_1d(x)
    outputs = np.empty((len(x), nfluxes)) if out is None else out

    # Outside of alpha bounds
    alpha_min, alpha_max = alphas[0], alphas[-1]
    idx_low = x < alpha_min
    idx_high = x > alpha_max
    co = np.zeros(3)

    for i in range(nfluxes):
        if idx_low.sum() > 0:
            # Linear trend in low extrapolation
            g = quad_grad(coefs[i, 0], alpha_min)
            co[2] = quad_fun(coefs[i, 0], alpha_min) - g * alpha_min
            co[1] = g
            co[0] = 0
            outputs[idx_low, i] = quad_fun(co, x[idx_low])

        if idx_high.sum() > 0:
            # Linear trend in high extrapolation
            g = quad_grad(coefs[i, -1], alpha_max)
            co[2] = quad_fun(coefs[i, -1], alpha_max) - g * alpha_max
            co[1] = g
            co[0] = 0
            outputs[idx_high, i] = quad_fun(co, x[idx_high])

    # Inside alpha bounds
    for j in range(nalphas-1):
        idx = (x >= alphas[j] - 1e-10) & (x <= alphas[j+ 1] + 1e-10)
        if idx.sum() == 0:
            continue

        for i in range(nfluxes):
            outputs[idx, i] = quad_fun(coefs[i, j], x[idx])

    return outputs
