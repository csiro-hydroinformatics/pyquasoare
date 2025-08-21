import warnings
import numpy as np

from pyquasoare import has_c_module
if has_c_module():
    import c_pyquasoare
else:
    raise ImportError("Cannot run quasoare without C module."
                      + " Please compile C code")

ERRORS = ["ignore", "raise", "warn"]


def quad_model(alphas, scalings,
               a_matrix_noscaling,
               b_matrix_noscaling,
               c_matrix_noscaling, s0, timestep,
               reset=None,
               errors="ignore",
               fluxes=None, niter=None, s1=None):
    """ Integrate the approximate reservoir equation with initial initial
    condition s0 over a series of P timesteps.

    The piecewise quadratic approximations of flux functions are defined by
    - interpolation nodes alphas
    - interpolation coefficients a_matrix_noscaling, b_matrix_noscaling,
        c_matrix_noscaling.

    The function allows scaling the fluxes by multiplicative factors
    given in the scalings matrix. The number of scaling factors should
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
    s0 : float
        Initial condition
    timestep : float
        Integration time step.
    reset : np.ndarray
        Timeseries indicating when to reset the model (=1).
    errors: str
        - if 'ignore', then skip over error during time step integration
        - if 'raise', throw a ValueError.
        - if 'warn', throw a warning.
    niter, s1, fluxes : np.ndarray
        Arrays used by for computation "in place", i.e.
        when output arrays are not allocated within the
        function. This is useful if the function is used repeatedly.

        If None, the arrays are allocated during every function run.

    Returns
    -------
    niter : np.ndarray
        Number of iterations (i.e. number of interpolation bands crossed).
        Vector of length P (one number for each timestep).
    s1 : np.ndarray
        Final value of S at the end of each timestep.
        Vector of length P.
    fluxes : np.ndarray
        Total of fluxes for each timestep.
        Matrix of size [P, M] (one value for each timestep and each flux).

    See Also
    --------
    approx.quad_fun : Approximation function.
    approx.quad_coefficient_matrix : Obtain interpolation coefficients.
    integrate.quad_forward : Integrate ODE forward.
    integrate.quad_integrate : Integrate ODE forward.

    Examples
    --------
    >>> from pyquasoare import approx, integrate, models
    >>> # Two fluxes : fixed inflow and outflow defined as a power function
    >>> funs = [lambda x: 1., lambda x: -x**6/2]
    >>> # 10 interpolation nodes
    >>> alphas = np.linspace(0., 1.2, 10)
    >>> # Get interpolation coefficients for each band and each flux
    >>> amat, bmat, cmat, cst = \
             approx.quad_coefficient_matrix(funs, alphas, approx_opt=1)
    >>> # Define a simulation with 10 time steps
    >>> P = 10
    >>> # The inflow is a step response equal to 0 excel for the 3rd time step
    >>> inflows = np.zeros(P)
    >>> inflows[2] = 100.
    >>> sc = np.column_stack([inflows, np.ones(P)])
    >>> # Run the model
    >>> s0 = 0.
    >>> timestep = 1.
    >>> niter, s1, fx = quad_model(alphas, sc, amat, bmat, cmat, s0, timestep)
    >>> # ODE solution
    >>> s1
    array([ 0.        ,  0.        , 14.63001824,  0.92614791,  0.75904346,
            0.68854374,  0.64494538,  0.61378862,  0.59006276,  0.57111566])
    >>> # Flux totals
    >>> fx
    array([[ 0.00000000e+00,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+02, -8.53699818e+01],
           [ 0.00000000e+00, -1.37038703e+01],
           [ 0.00000000e+00, -1.67104449e-01],
           [ 0.00000000e+00, -7.04997259e-02],
           [ 0.00000000e+00, -4.35983531e-02],
           [ 0.00000000e+00, -3.11567661e-02],
           [ 0.00000000e+00, -2.37258597e-02],
           [ 0.00000000e+00, -1.89470944e-02]])
    >>> # Check mass balance
    >>> fx.sum(axis=0)[1]+s0-s1[-1]
    -100.0
    """
    assert errors in ERRORS
    nval = scalings.shape[0]

    if fluxes is None:
        fluxes = np.zeros(scalings.shape, dtype=np.float64)
    if niter is None:
        niter = np.zeros(nval, dtype=np.int32)
    if s1 is None:
        s1 = np.zeros(nval, dtype=np.float64)
    if reset is None:
        reset = np.zeros(nval, dtype=np.int32)
    else:
        reset = reset.astype(np.int32)

    ierrors = np.int32(ERRORS.index(errors))

    ierr = c_pyquasoare.quad_model(ierrors, alphas, scalings,
                                   reset,
                                   a_matrix_noscaling,
                                   b_matrix_noscaling,
                                   c_matrix_noscaling,
                                   s0, timestep, niter, s1, fluxes)

    if errors == "raise" and ierr > 0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.quad_model returns {ierr} ({mess})")

    if errors == "warn":
        if np.any(niter < 0):
            nerr = (niter < 0).sum()
            mess = f"{nerr} errors when running c_pyquasoare.quad_model"
            warnings.warn(mess)

    return niter, s1, fluxes
