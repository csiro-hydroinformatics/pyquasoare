from collections import namedtuple
import numpy as np

from pyquasoare import has_c_module, approx
from pyquasoare.models import errors_check, error_management, get_errorsid

if has_c_module():
    import c_pyquasoare_grad
else:
    raise ImportError("Cannot run quasoare without C module."
                      + " Please compile C code")


NFLUXES_POTENTIAL = c_pyquasoare_grad.NFLUXES_POTENTIAL
NFLUXES_INPUTS = c_pyquasoare_grad.NFLUXES_INPUTS
NFLUXES_OUTPUTS = c_pyquasoare_grad.NFLUXES_OUTPUTS
INPUT_MAPPING_LASTDIM = c_pyquasoare_grad.INPUT_MAPPING_LASTDIM
INPUT_MAPPING_SCALINGFACTOR = c_pyquasoare_grad.INPUT_MAPPING_SCALINGFACTOR
VARTYPE_NUMBER = c_pyquasoare_grad.VARTYPE_NUMBER

IFLUX_INFLOW = c_pyquasoare_grad.IFLUX_INFLOW
IFLUX_DEMAND = c_pyquasoare_grad.IFLUX_DEMAND
IFLUX_OUTFLOW = c_pyquasoare_grad.IFLUX_OUTFLOW
IFLUX_BYPASS = c_pyquasoare_grad.IFLUX_BYPASS
IFLUX_SHORTFALL = c_pyquasoare_grad.IFLUX_SHORTFALL
IFLUX_TOTALOUT = c_pyquasoare_grad.IFLUX_TOTALOUT


def lin_coefficient_matrix(funs, alphas):
    if len(funs) != NFLUXES_INPUTS:
        raise ValueError(f"Expected {NFLUXES_INPUTS} flux functions, got {len(funs)}.")
    coefs = approx.quad_coefficient_matrix(funs, alphas,
                                           approx_opt=0)
    return np.ascontiguousarray(coefs[:, :, 1:])

NetworkLink = namedtuple("Link",
                         [
                         "upstream_reservoir_id",
                         "upstream_output_flux_id",
                         "downstream_reservoir_id",
                         "downstream_potential_flux_id",
                         "weight"
                         ])

class Network():
    def __init__(self, nreservoirs):
        self.nreservoirs = nreservoirs
        self.links = []

    @property
    def input_mapping(self):
        nin = NFLUXES_POTENTIAL
        nout = NFLUXES_OUTPUTS
        nres = self.nreservoirs
        nipmd = INPUT_MAPPING_LASTDIM
        sc = INPUT_MAPPING_SCALINGFACTOR

        ipm = np.empty((nres, nres * nout, nipmd), dtype=np.int32)
        ipm.fill(-1)

        for lk in self.links:
            rdid = lk.downstream_reservoir_id
            idid = lk.downstream_potential_flux_id
            ruid = lk.upstream_reservoir_id
            ouid = lk.upstream_output_flux_id

            lks = ipm[rdid]
            ilk = np.where(lks[:, 0] < 0)[0][0]
            if ilk >= ipm.shape[1]:
                errmsg = "Exceeded the maximum number of inter-reservoir links"
                raise ValueError(errmsg)
            lks[ilk, :] = [ruid, ouid, idid, int(lk.weight * sc)]

        return ipm

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

    def add_link(self, upstream_reservoir_id, upstream_output_flux_id,
                 downstream_reservoir_id, downstream_potential_flux_id,
                 weight):
        nres = self.nreservoirs

        # Check data
        upid = int(upstream_reservoir_id)
        if upid >= nres or upid < 0:
            errmsg = "Expected upstream reservoir id in"\
                     + f" [0, {nres}-1], got {upid}"
            raise ValueError(errmsg)

        upoid = int(upstream_output_flux_id)
        if upoid >= NFLUXES_OUTPUTS or upoid < 0:
            errmsg = "Expected upstream reservoir output id in"\
                     + f" [0, {NFLUXES_OUTPUTS}-1], got {upoid}"
            raise ValueError(errmsg)

        downid = int(downstream_reservoir_id)
        if downid >= nres or downid < 0:
            errmsg = "Expected downstream reservoir id in"\
                     + f" [0, {nres}-1], got {downid}"
            raise ValueError(errmsg)

        downiid = int(downstream_potential_flux_id)
        if downiid >= NFLUXES_POTENTIAL or downiid < 0:
            errmsg = "Expected downstream reservoir input id in"\
                     + f" [0, {NFLUXES_POTENTIAL}-1], got {downiid}"
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
                               downstream_reservoir_id,
                               conversion_factor=1.):
        """ Links total outflow from upstream to inflow for downstream.
        Weight is -1 because we need to convert the outflow (<0) from
        upstream reservoir to positive inflow to the downstream reservoir.
        """
        if conversion_factor <= 0:
            errmsg = "Expected conversion_factor > 0."
            raise ValueError(errmsg)

        self.add_link(upstream_reservoir_id, IFLUX_TOTALOUT,
                      downstream_reservoir_id, IFLUX_INFLOW,
                      -conversion_factor)

    def add_link_bypass_shortfall(self, upstream_reservoir_id,
                                  downstream_reservoir_id,
                                  conversion_factor=1.):
        """ Links upstram inflow bypass to downstream inflow and
        upstream demand shortfall to downstream demand.
        Weight is -1 because we need to convert the outflow (<0) from
        upstream reservoir to positive inflow to the downstream reservoir.
        """
        if conversion_factor <= 0:
            errmsg = "Expected conversion_factor > 0."
            raise ValueError(errmsg)

        self.add_link(upstream_reservoir_id, IFLUX_BYPASS,
                      downstream_reservoir_id, IFLUX_INFLOW,
                      -conversion_factor)

        self.add_link(upstream_reservoir_id, IFLUX_SHORTFALL,
                      downstream_reservoir_id, IFLUX_DEMAND,
                      -conversion_factor)


def grad_reservoir(alphas, gamma,
                   x0,
                   Qstars,
                   coefs_linear,
                   timestep=1.,
                   dx0=None,
                   flux_scalings=None,
                   x1=None,
                   dx1=None,
                   fluxes=None,
                   dfluxes=None,
                   niter=None,
                   errors="ignore"):
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
    gamma : float
        Reservoir capacity
    x0 : float
        Initial condition
    Qstars : np.ndarray
        Timeseries of the 2 potential fluxes.
        This is a 2D array of dimension [N X 2] with N the number of timesteps.
    coefs_linear: np.ndarray
        Linear interpolation coefficients for the 3 flux functions and each
        interpolation band. This is a 3D array of size [3 x M-1 x 2].
        There are 2 coefficients (linear interpolation) for each of the 3 input
        fluxes and each  of the M-1 interpolation bands (see argument 'alphas').
    timestep : float
        Integration time step.
    dx0 : np.array
        Initial derivative condition against independent variables.
        This is a vector of length VARTYPE_NUMBER.
        Here we assume that we seek the derivative of the output fluxes
        against all possible types of  independent variables.
        By default, this vector is initialised to an vector of 0.
    flux_scalings : np.array
        Scaling factors applied to each flux.
        This is a vector of size NFLUXES_INPUTS.
        By default, this vector is initialised to an vector of 1.
    fluxes, dfluxes, x1, dx1 : np.ndarray
        Arrays used by for computation "in place", i.e.
        when output arrays are not allocated within the
        function. This is useful if the function is used repeatedly.
        If None, the arrays are allocated during every function run.
    errors: str
        - if 'ignore', then skip over error during time step integration
        - if 'raise', throw a ValueError.
        - if 'warn', throw a warning.

    Returns
    -------
    niter : np.ndarray
        Number of iterations (i.e. number of interpolation bands crossed).
        Vector of length P (one number for each timestep).
    x1 : np.ndarray
        Final value of x at the end of each timestep.
        Vector of length P.
    dx1 : np.ndarray
        Derivative of x1 against each independent variable at the end of each timestep.
        Array of size [P, Q] where Q is the number of independent variables
        (see dx0 argument above).
    fluxes : np.ndarray
        Integrated output fluxes for each timestep.
        Matrix of size [P, NO] (one value for each timestep and each flux)
        where NO is the number of outputs fluxes (constant NFLUXES_OUTPUTS).
    dfluxes : np.ndarray
        Derivative of the integrated output fluxes for each timestep.
        Array of size [P, Q, NO] (one value for each timestep and each flux)
        where NO is the number of outputs fluxes (constant NFLUXES_OUTPUTS).

    See Also
    --------
    approx.quad_fun : Approximation function.
    approx.quad_coefficient_matrix : Obtain interpolation coefficients.
    integrate.quad_forward : Integrate ODE forward.
    integrate.quad_integrate : Integrate ODE forward.
    """
    errors_check(errors)
    nval = Qstars.shape[0]
    nvariables = VARTYPE_NUMBER

    if dx0 is None:
        dx0 = np.zeros(nvariables, dtype=np.float64)

    if flux_scalings is None:
        flux_scalings = np.ones(3)

    if niter is None:
        niter = np.zeros(nval, dtype=np.int32)

    if x1 is None:
        x1 = np.zeros(nval, dtype=np.float64)

    if dx1 is None:
        dx1 = np.zeros((nval, nvariables), dtype=np.float64)

    if fluxes is None:
        fluxes = np.zeros((nval, NFLUXES_OUTPUTS), dtype=np.float64)

    if dfluxes is None:
        dfluxes = np.zeros((nval, nvariables, NFLUXES_OUTPUTS),
                           dtype=np.float64)

    ierrors = get_errorsid(errors)
    ierr = c_pyquasoare_grad.grad_reservoir(ierrors,
                                            alphas,
                                            timestep,
                                            gamma,
                                            flux_scalings,
                                            x0,
                                            dx0,
                                            Qstars,
                                            coefs_linear,
                                            niter,
                                            x1,
                                            dx1,
                                            fluxes,
                                            dfluxes)
    name = "c_pyquasoare_grad.grad_reservoir"
    error_management(errors, ierr, name)

    return niter, x1, dx1, fluxes, dfluxes


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
