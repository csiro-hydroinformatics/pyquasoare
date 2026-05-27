import warnings
import numpy as np

from pyquasoare import has_c_module, approx
from pyquasoare.models import ERRORS

if has_c_module():
    import c_pyquasoare
else:
    raise ImportError("Cannot run quasoare without C module."
                      + " Please compile C code")

def lin_coefficient_matrix(funs, alphas):
    if len(funs) != 3:
        raise ValueError(f"Expected 3 flux functions, got {len(funs)}.")
    coefs = approx.quad_coefficient_matrix(funs, alphas,
                                           approx_opt=0)
    return np.ascontiguousarray(coefs[:, :, 1:])



def lin_reservoir(alphas, timestep, gamma,
                  Qstars, coefs_linear, x0,
                  xmin=-np.inf,
                  xmax=np.inf,
                  errors="ignore",
                  fluxes=None, niter=None, x1=None):
    """ Integrate the approximate reservoir equation with initial initial
    condition x0 over a series of P timesteps. The reservoir is
    characterised by the following differential equation:

    dx/dt = Qstar1 / Γ × f1(x) + Qstar2 × Γ * f2(x) - Γ / Δ × f3(x)

    where Qstar1 (inflow) and Qstar2 (demand) are potential fluxes, Γ is the reservoir
    capacity and Δ is the timestep, and fi are flux functions that
    are assumed to be positive.

    The output integrated fluxes are defined as

    Q[i] = Qstar[i] × integral( fi(x), t=0 .. Δ)

    Hence, 3 output fluxes are defined in this way. In addition, 3 other fluxes
    correspond to:
    Q[4] = Qstar[1] - Q[1]   # Inflow bypass around the reservoir
    Q[5] = Qstar[2] - Q[2]   # Demand shortfall
    Q[6] = - Q[4] + Q[3]       # Total outflow


    The flux functions fi are approximated by piecewise linear
    interpolation functions that are defined by
    - interpolation nodes alphas
    - interpolation coefficients coefs_linear

    Parameters
    ----------
    alphas : np.ndarray
        Approximation nodes. Vector of length M.
        Should be strictly increasing, i.e. alphas[i] > alphas[i-1]
    Qstars : np.ndarray
        Timeseries of the 2 potential fluxes.
        This is a 2D array of dimension [N X 2] with N the number of timesteps.
    coefs_linear: np.ndarray
        Linear interpolation coefficients for the 3 flux functions and each
        interpolation band. This is a 3D array of size [3 x M-1 x 2].
        There are 2 coefficients for each flux and each band corresponding
        to the two linear interpolation terms.
    x0 : float
        Initial condition
    xmin : float
        Minimal store value.
    xmax : float
        Maximum store value.
    timestep : float
        Integration time step.
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
    x1 : np.ndarray
        Final value of x at the end of each timestep.
        Vector of length P.
    fluxes : np.ndarray
        Value of the 6 integrated output fluxes for each timestep.
        Matrix of size [P, 6] (one value for each timestep and each flux).

    See Also
    --------
    approx.quad_fun : Approximation function.
    approx.quad_coefficient_matrix : Obtain interpolation coefficients.
    integrate.quad_forward : Integrate ODE forward.
    integrate.quad_integrate : Integrate ODE forward.
    """
    if errors not in ERRORS:
        txt = "/".join(ERRORS)
        errmsg = f"Expected errors in {txt}, got {errors}."
        raise ValueError(errmsg)

    nval = Qstars.shape[0]

    if fluxes is None:
        fluxes = np.zeros((nval, 6), dtype=np.float64)
    if niter is None:
        niter = np.zeros(nval, dtype=np.int32)
    if x1 is None:
        x1 = np.zeros(nval, dtype=np.float64)

    ierrors = np.int32(ERRORS.index(errors))

    ierr = c_pyquasoare.lin_reservoir(ierrors, timestep, gamma,
                                      x0, xmin, xmax,
                                      alphas,
                                      Qstars,
                                      coefs_linear,
                                      niter,
                                      x1,
                                      fluxes)

    if errors == "raise" and ierr > 0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.lin_model returns {ierr} ({mess})")

    if errors == "warn":
        if np.any(niter < 0):
            nerr = (niter < 0).sum()
            mess = f"{nerr} errors when running c_pyquasoare.lin_model"
            warnings.warn(mess)

    return niter, x1, fluxes
