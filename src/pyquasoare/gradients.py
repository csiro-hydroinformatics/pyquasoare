from collections import namedtuple
import numpy as np

from pyquasoare import has_c_module, approx
from pyquasoare.models import errors_check, error_management, get_errorsid

NPOTENTIAL_FLUXES = 2
NOUTPUT_FLUXES = 6

if has_c_module():
    import c_pyquasoare_grad
else:
    raise ImportError("Cannot run quasoare without C module."
                      + " Please compile C code")


def lin_coefficient_matrix(funs, alphas):
    if len(funs) != 3:
        raise ValueError(f"Expected 3 flux functions, got {len(funs)}.")
    coefs = approx.quad_coefficient_matrix(funs, alphas,
                                           approx_opt=0)
    return np.ascontiguousarray(coefs[:, :, 1:])

NetworkLink = namedtuple("Link",
                         [
                         "upstream_reservoir_id",
                         "upstream_output_id",
                         "downstream_reservoir_id",
                         "downstream_input_id",
                         "weight"
                         ])

class Network():
    def __init__(self, nreservoirs):
        self.nreservoirs = nreservoirs
        self.links = []

    @property
    def Qstars_mapping(self):
        nin = NPOTENTIAL_FLUXES
        nout = NOUTPUT_FLUXES
        nres = self.nreservoirs

        qstm = np.empty((nres * nin, nres * nout))
        qstm.fill(np.nan)

        for lk in self.links:
            i = nin * lk.downstream_reservoir_id + lk.downstream_input_id
            j = nout * lk.upstream_reservoir_id + lk.upstream_output_id
            qstm[i, j] = lk.weight

        return qstm

    def upstream_reservoirs(self, reservoir_id):
        # Recursive function
        def add_upids(rid, upids):
            upids.append(rid)
            for lk in self.links:
                if lk.downstream_reservoir_id == rid:
                    upid = lk.upstream_reservoir_id

                    if len(upids) > 0:
                        if upid == upids[0]:
                            errmsg = "Reservoir receives"\
                                     + " inputs from itself. This is"\
                                     + " cyclical reservoir connection."
                            raise ValueError(errmsg)

                    # Move upstream
                    upids = add_upids(upid, upids)

            return upids

        return set(add_upids(reservoir_id, [])[1:])

    def add_link(self, upstream_reservoir_id, upstream_output_id,
                 downstream_reservoir_id, downstream_input_id,
                 weight):
        nres = self.nreservoirs

        # Check data
        upid = int(upstream_reservoir_id)
        if upid >= nres or upid < 0:
            errmsg = "Expected upstream reservoir id in"\
                     + f" [0, {nres}-1], got {upid}"
            raise ValueError(errmsg)

        upoid = int(upstream_output_id)
        if upoid >= NOUTPUT_FLUXES or upoid < 0:
            errmsg = "Expected upstream reservoir output id in"\
                     + f" [0, {NOUTPUT_FLUXES}-1], got {upoid}"
            raise ValueError(errmsg)

        downid = int(downstream_reservoir_id)
        if downid >= nres or downid < 0:
            errmsg = "Expected downstream reservoir id in"\
                     + f" [0, {nres}-1], got {downid}"
            raise ValueError(errmsg)

        downiid = int(downstream_input_id)
        if downiid >= NPOTENTIAL_FLUXES or downiid < 0:
            errmsg = "Expected downstream reservoir input id in"\
                     + f" [0, {NPOTENTIAL_FLUXES}-1], got {downiid}"
            raise ValueError(errmsg)

        if downid == upid:
            errmsg = "Expected downstream reservoir id to "\
                     + " different from upstream reservoir id"
            raise ValueError(errmsg)

        # Create link
        lk = NetworkLink(upid, upoid, downid, downiid, weight)
        self.links.append(lk)

        # check non-cyclic
        self.upstream_reservoirs(downid)

    def add_link_total_outflow(self, upstream_reservoir_id,
                               downstream_reservoir_id):
        """ Links total outflow from upstream to inflow for downstream.
        """
        self.add_link(upstream_reservoir_id, 5,
                      downstream_reservoir_id, 0, -1.)

    def add_link_bypass_shortfall(self, upstream_reservoir_id,
                                  downstream_reservoir_id):
        """ Links upstram inflow bypass to downstream inflow and
        upstream demand shortfall to downstream demand.
        """
        self.add_link(upstream_reservoir_id, 3,
                      downstream_reservoir_id, 0, -1.)
        self.add_link(upstream_reservoir_id, 4,
                      downstream_reservoir_id, 1, -1.)


def grad_reservoir(alphas, timestep,
                  Qstars, gamma, coefs_linear,
                  x0, dx0, variables_type_id,
                  current_reservoir_id=0,
                  variables_reservoir_id=None,
                  xmin=-np.inf, xmax=np.inf,
                  flux_scalings=None,
                  errors="ignore",
                  fluxes=None, niter=None, x1=None, dx1=None):
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
    gamma : float
        Reservoir capacity
    coefs_linear: np.ndarray
        Linear interpolation coefficients for the 3 flux functions and each
        interpolation band. This is a 3D array of size [3 x M-1 x 2].
        There are 2 coefficients for each flux and each band corresponding
        to the two linear interpolation terms.
    x0 : float
        Initial condition
    dx0 : np.array
        Initial derivative condition
    variables_type_id : np.array
        Type of independent variables
    xmin : float
        Minimal store value.
    xmax : float
        Maximum store value.
    timestep : float
        Integration time step.
    flux_scalings : np.array
        Scaling factors applied to each flux. Default is 1.
    errors: str
        - if 'ignore', then skip over error during time step integration
        - if 'raise', throw a ValueError.
        - if 'warn', throw a warning.
    fluxes, niter, x1, dx1 : np.ndarray
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
    dx1 : np.ndarray
        Derivative of x1 against capacity Γ at the end of each timestep.
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
    errors_check(errors)
    nval = Qstars.shape[0]
    nvariables = len(dx0)

    if variables_reservoir_id is None:
        variables_reservoir_id = current_reservoir_id \
            * np.ones(nvariables, dtype=np.int32)
    if flux_scalings is None:
        flux_scalings = np.ones(3)
    if fluxes is None:
        fluxes = np.zeros((nval, 6), dtype=np.float64)
    if niter is None:
        niter = np.zeros(nval, dtype=np.int32)
    if x1 is None:
        x1 = np.zeros(nval, dtype=np.float64)
    if dx1 is None:
        dx1 = np.zeros(nval, dtype=np.float64)

    ierrors = get_errorsid(errors)
    ierr = c_pyquasoare_grad.grad_reservoir(ierrors, timestep, gamma,
                                            flux_scalings,
                                            xmin, xmax, x0,
                                            dx0,
                                            current_reservoir_id,
                                            variables_reservoir_id,
                                            variables_type_id,
                                            alphas,
                                            Qstars,
                                            coefs_linear,
                                            niter,
                                            x1,
                                            dx1,
                                            fluxes)
    name = "c_pyquasoare_grad.grad_reservoir"
    error_management(errors, ierr, name)

    return niter, x1, dx1, fluxes


def grad_model(alphas, timestep,
               Qstars, Qstars_mapping,
               gammas, coefs_linear, x0,
               variables_reservoir_id=None,
               fluxes=None, x1=None, dx1=None,
               flux_scalings=None,
               errors="ignore"):
    nreservoirs = alphas.shape[0]
    nval = Qstars.shape[0]

    if variables_reservoir_id is None:
        variables_reservoir_id = np.arange(nreservoirs, dtype=np.int32)
    nvariables = variables_reservoir_id.shape[0]

    dx0 = np.zeros(nvariables, dtype=np.float64)

    if flux_scalings is None:
        flux_scalings = np.ones(3 * nreservoirs)
    if fluxes is None:
        fluxes = np.zeros((nval, 6 * nreservoirs), dtype=np.float64)
    if x1 is None:
        x1 = np.zeros((nval, nreservoirs), dtype=np.float64)
    if dx1 is None:
        dx1 = np.zeros((nval, nreservoirs, nvariables), dtype=np.float64)

    ierr = c_pyquasoare_grad.grad_model(errors, timestep, x0,
                                        dx0,
                                        variables_reservoir_id,
                                        alphas,
                                        Qstars, Qstars_mapping,
                                        gammas,
                                        coefs_linear,
                                        x1, fluxes)
    name = "c_pyquasoare_grad.grad_model"
    error_management(errors, ierr, name)

    return x1, fluxes
