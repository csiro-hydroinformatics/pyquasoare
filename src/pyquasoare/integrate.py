import numpy as np

from pyquasoare import has_c_module, approx
from pyquasoare.approx import QUASOARE_NFLUXES_MAX

if has_c_module():
    import c_pyquasoare
else:
    raise ImportError("Cannot run quasoare without C code. Please compile C code.")


def __find_alpha(alphas, u0):
    """ -- Function defined for test purposes --
    Finds location of u0 within interpolation bands defined by
    the interpolation nodes alphas.
    """
    return c_pyquasoare.find_alpha(alphas, u0)


def __eta_fun(x, Delta):
    """ -- Function defined for test purposes --
    Compute the eta function equal either to atan(x) if Delta<0, atanh(x)
    if Delta>=0 and |x|<1 or atanh(1/x) if Delta>0 and |x|>1.
    """
    return c_pyquasoare.eta_fun(x, Delta)


def quad_constants(a, b, c):
    constants = np.zeros(3)
    ierr = c_pyquasoare.quad_constants(a, b, c, constants)
    if ierr>0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.constants returns {ierr} ({mess})")
    return constants


def quad_forward(a, b, c, Delta, qD, sbar, t0, s0, t):
    """ Forward integration of approximated reservoir equation defined by
    dS/dt = a.S^2+b.S+c using analytical solution.

    Parameters
    ----------
    a, b, c : float
        Equation parameters
    Delta, qD, sbar : float
        Constants derived from parameters:
        Delta = b^2-4ac
        qD = sqrt(|Delta|)/2
        sbar = -b/2a
    t0 : float
        Start time
    s0 : float
        Initial condition
    t : float or np.ndarray
        End time(s)

    Return
    ------
    s : float or np.ndarray
        Solution s(t)

    See Also
    --------
    approx.quad_coefficients : Compute approximation coefficients.
    quad_constant : Compute constants.

    Examples
    --------
    >>> from pyquasoare import approx, integrate
    >>> fun = lambda x: 1-x**6/2
    >>> a0, a1 = 0.6, 1.2
    >>> f0, f1, fm = fun(a0), fun(a1), fun((a0+a1)/2)
    >>> a, b, c = approx.quad_coefficients(a0, a1, f0, f1, fm, approx_opt=1)
    >>> Delta, qD, sbar = integrate.quad_constants(a, b, c)
    >>> t0 = 0; t = np.linspace(t0, 1, 10); s0 = 0.8
    >>> quad_forward(a, b, c, Delta, qD, sbar, t0, s0, t)
    array([0.8       , 0.88163417, 0.945038  , 0.99152036, 1.0241747 ,
           1.04643346, 1.06129603, 1.07108369, 1.07747071, 1.0816138 ])
    """
    if np.isscalar(t):
        return c_pyquasoare.quad_forward(a, b, c, Delta, qD, sbar, t0, s0, t)
    else:
        s = np.nan*np.ones_like(t)
        ierr = c_pyquasoare.quad_forward_vect(a, b, c, Delta, qD, sbar, t0, s0, t, s)
        if ierr>0:
            raise ValueError(f"c_pyquasoare.quad_forward_vect returns {ierr}")
        return s


def quad_delta_t_max(a, b, c, Delta, qD, sbar, s0):
    """ Maximum time for which the solution of the following ODE remains
    valid:
    dS/dt = a.S^2+b.S+c
    with initial time t0=0 and initial condition s(t0)=s0.

    Parameters
    ----------
    a, b, c : float
        Equation parameters
    Delta, qD, sbar : float
        Constants derived from parameters:
        Delta = b^2-4ac
        qD = sqrt(|Delta|)/2
        sbar = -b/2a
    s0 : float
        Initial condition

    Returns
    -------
    float
        Maximum time for which s(t) remains valid.
        Can be +infty (i.e. s(t) is always valid for t>0).

    See Also
    --------
    quad_constant : Compute constants.

    Examples
    --------
    >>> from pyquasoare import approx, integrate
    >>> fun = lambda x: 1-x**6/2
    >>> a0, a1 = 0.6, 1.2
    >>> f0, f1, fm = fun(a0), fun(a1), fun((a0+a1)/2)
    >>> a, b, c = approx.quad_coefficients(a0, a1, f0, f1, fm, approx_opt=1)
    >>> Delta, qD, sbar = integrate.quad_constants(a, b, c)
    >>> quad_delta_t_max(a, b, c, Delta, qD, sbar, s0=0.8)
    inf
    >>> quad_delta_t_max(a, b, c, Delta, qD, sbar, s0=-2.)
    0.09534868597457016
    """
    return c_pyquasoare.quad_delta_t_max(a, b, c, Delta, qD, sbar, s0)


def quad_inverse(a, b, c, Delta, qD, sbar, s0, s1):
    """ Compute time when solution of the following ODE will reach s1
    dS/dt = a.S^2+b.S+c
    with initial time t0=0 and initial condition s(t0)=s0.

    Parameters
    ----------
    a, b, c : float
        Equation parameters
    Delta, qD, sbar : float
        Constants derived from parameters:
        Delta = b^2-4ac
        qD = sqrt(|Delta|)/2
        sbar = -b/2a
    s0 : float
        Initial condition
    s1 : float
        Target value.

    Returns
    -------
    float
        Time for which s(t) = s1.
        Can be nan if value is never reached.

    See Also
    --------
    quad_constant : Compute constants.

    Examples
    --------
    >>> from pyquasoare import approx, integrate
    >>> fun = lambda x: 1-x**6/2
    >>> a0, a1 = 0.6, 1.2
    >>> f0, f1, fm = fun(a0), fun(a1), fun((a0+a1)/2)
    >>> a, b, c = approx.quad_coefficients(a0, a1, f0, f1, fm, approx_opt=1)
    >>> Delta, qD, sbar = integrate.quad_constants(a, b, c)
    >>> quad_inverse(a, b, c, Delta, qD, sbar, s0=0.8, s1=1.)
    0.3584916286499921
    """
    if np.isscalar(s1):
        return c_pyquasoare.quad_inverse(a, b, c, Delta, qD, sbar, s0, s1)
    else:
        t = np.nan*np.ones_like(s1)
        ierr = c_pyquasoare.quad_inverse_vect(a, b, c, Delta, qD, sbar, s0, s1, t)
        if ierr>0:
            raise ValueError(f"c_pyquasoare.quad_inverse_vect returns {ierr}")
        return t


def quad_fluxes(a_vector, b_vector, c_vector, \
                        Aj, Bj, Cj, \
                        Delta, qD, sbar, \
                        t0, t1, s0, s1, fluxes):
    """ Increment flux totals during integration of the approximate reservoir
    equation
    dS/dt = Aj.S^2+Bj.S+Cj
    with initial time t0 and initial condition s(t0)=s0. The integration
    is undertaken up to t=t1, reaching a final value s(t1) = s1. The value
    of s1 is computed with the quad_forward function.

    The fluxes are defined as quadratic functions
    fi(S) = aij.S^2+bij.S+cij
    aij are stored in the vector a_vector, bij in b_vector and
    cij in c_vector.

    The coefficients Aj, Bj, Cj are coefficients sums:
    Aj = sum(aij, i)
    Bj = sum(bij, i)
    Cj = sum(cij, i)

    Note that the function does not return anything. The flux
    vector is incremented in place.

    Parameters
    ----------
    a_vector, b_vector, c_vector : float
        Flux interpolation coefficients
    Delta, qD, sbar : float
        Constants derived from parameters:
        Delta = Bj^2-4.Aj.Cj
        qD = sqrt(|Delta|)/2
        sbar = -Bj/2Aj
    t0, t1 : float
        Initial and end times.
    s0, s1 : float
        Initial and final condition
    fluxes : np.ndarray
        Flux vector to be incremented.

    See Also
    --------
    quad_constant : Compute constants.
    quad_forward : Integrate ODE forward.
    """

    ierr = c_pyquasoare.quad_fluxes(a_vector, b_vector, c_vector, \
                            Aj, Bj, Cj, Delta, qD, sbar, \
                            t0, t1, s0, s1, fluxes)
    if ierr>0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.quad_fluxes returns {ierr} ({mess})")


def quad_integrate(alphas, scalings, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling, \
                t0, s0, timestep):
    """ Integrate the approximate reservoir equation with initial time t0
    and initial condition s(t0)=s0 over a single timestep.

    The piecewise quadratic approximations of flux functions are defined by
    - interpolation nodes alphas
    - interpolation coefficients a_matrix_noscaling, b_matrix_noscaling,
        c_matrix_noscaling.

    The function allows scaling the fluxes by multiplicative factors
    given in the scalings vector. The number of scaling factors should
    be identical to the number of fluxes (hence the number of columns
    in a_matrix_noscaling, b_matrix_noscaling and c_matrix_noscaling).

    Parameters
    ----------
    alphas : np.ndarray
        Approximation nodes. Vector of length M.
        Should be strictly increasing, i.e. alphas[i]>alphas[i-1]
    scalings : np.ndarray
        Vector of scaling factors applied to fluxes (array of size N).
    a_matrix_noscaling, b_matrix_noscaling, c_matrix_noscaling: np.ndarray
        Interpolation coefficients for each flux and each
        interpolation band (i.e. matrices of size [M-1, N]).
    t0 : float
        Start time.
    s0 : float
        Initial condition
    timestep : float
        Integration time step.

    Returns
    -------
    niter : int
        Number of iterations (i.e. number of interpolation bands crossed).
    s1 : float
        Final value of S at t=t1.
    fluxes : np.ndarray
        Total of fluxes between t=t0 and t=t1.

    See Also
    --------
    approx.quad_fun : Approximation function.
    approx.quad_coefficient_matrix : Obtain interpolation coefficients.
    quad_forward : Integrate ODE forward.

    Examples
    --------
    >>> from pyquasoare import approx, integrate
    >>> fun = lambda x: 1-x**6/2
    >>> alphas = np.array([0.6, 0.8, 1.2])
    >>> amat, bmat, cmat, cst = approx.quad_coefficient_matrix([fun], alphas, approx_opt=1)
    >>> sc = np.ones(1)
    >>> t0 = 0
    >>> s0 = 0.8
    >>> timestep = 1.
    >>> niter, s1, fx = quad_integrate(alphas, sc, amat, bmat, cmat, t0, s0, timestep)
    >>> niter
    1
    >>> s1
    1.1127726488382736
    >>> fx
    array([0.31277265])
    """
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


