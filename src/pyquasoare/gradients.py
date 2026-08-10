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
INPUT_MAPPING_MISSINGRESERVOIRID = c_pyquasoare_grad.INPUT_MAPPING_MISSINGRESERVOIRID
VARTYPE_NUMBER = c_pyquasoare_grad.VARTYPE_NUMBER

IFLUX_INFLOW = c_pyquasoare_grad.IFLUX_INFLOW
IFLUX_DEMAND = c_pyquasoare_grad.IFLUX_DEMAND
IFLUX_DRAINAGE = c_pyquasoare_grad.IFLUX_DRAINAGE
IFLUX_BYPASS = c_pyquasoare_grad.IFLUX_BYPASS
IFLUX_SHORTFALL = c_pyquasoare_grad.IFLUX_SHORTFALL
IFLUX_OUTFLOW = c_pyquasoare_grad.IFLUX_OUTFLOW

def thresh_smooth(x, eps):
    return  np.maximum(x, 0) / np.sqrt(x**2 + eps**2)

FLUX_FUNCTIONS = {
    "transluent": [
        lambda x: x * 0.,
        lambda x: x * 0.,
        lambda x: x * 0.
        ],
    "gr4jprod": [
        lambda x: 1 - np.clip(x, 0, 1)**2,
        lambda x: np.clip(x, 0, 1) * (2 - np.clip(x, 0, 1)),
        lambda x: 4**3/9**4 * np.maximum(x, 0)**5
        ],
    "gr2mprod": [
        lambda x: 1 - np.clip(x, 0, 1)**2,
        lambda x: np.clip(x, 0, 1) * (2 - np.clip(x, 0, 1)),
        lambda x: np.maximum(x, 0)**4 / 3.
        ],
    "gr4jroute": [
        # Safety on gr4j route store to avoid blow up if demand > 0
        lambda x: 0.9 * thresh_smooth(1 - x, 1e-3) * thresh_smooth(x , 1e-3),
        lambda x: np.maximum(x, 0.)**3.5,
        lambda x: np.maximum(x, 0.)**5 / 4
        ],
    "gr2mroute": [
        # Safety on gr2m route store to avoid blow up if demand > 0
        lambda x: thresh_smooth(1 - x, 1e-3) * thresh_smooth(x , 1e-3),
        lambda x: np.maximum(x, 0.),
        lambda x: np.maximum(x, 0.)**2
        ],
    "quadrouting": [
        lambda x: np.where(x > 0., 1., 0.),
        lambda x: np.ones_like(x),
        lambda x: np.maximum(x, 0.)**2
        ],
    "linrouting": [
        lambda x: np.where(x > 0., 1., 0.),
        lambda x: np.ones_like(x),
        lambda x: np.maximum(x, 0.)
        ],
    "openwater": [
        lambda x: np.where(x < 1., 1., 0.),
        lambda x: np.where(x > 0., 1., 0.),
        lambda x: np.zeros_like(x)
        ],
    "openwatersmooth": [
        lambda x: thresh_smooth(1 - x, 1e-2),
        lambda x: thresh_smooth(x, 1e-2),
        lambda x: np.zeros_like(x)
        ]
}


def get_alphas(nalphas, amin=0, amax=1.,
               nreservoirs=1,
               eps=1e-5):
    if eps <= 0:
        errmsg = f"Expected eps > 0, got {eps:0.1e}"
        raise ValueError(errmsg)

    a0 = amin + 2 * eps
    a1 = amax - 2 * eps
    if a0 >= a1:
        errmsg = f"Expected amin + 2eps < amax - 2eps, got {a0:0.1e} / {a1:0.2e}"
        raise ValueError(errmsg)

    if nalphas < 8:
        errmsg = f"Expected nalphas >= 8, got {nalphas}"
        raise ValueError(errmsg)

    alphas = [amin - eps, amin, amin + eps] \
             + np.linspace(a0, a1, nalphas - 6).tolist() \
             + [amax - eps, amax, amax + eps]
    return np.repeat(np.array(alphas)[None, :], nreservoirs, axis=0)


def lin_coefficient_matrix(funs, alphas):
    if len(funs) != NFLUXES_INPUTS:
        raise ValueError(f"Expected {NFLUXES_INPUTS} flux functions, got {len(funs)}.")
    coefs = approx.quad_coefficient_matrix(funs, alphas,
                                           approx_opt=0)
    return np.ascontiguousarray(coefs[:, :, 1:])


def grad_reservoir(alphas, gamma,
                   Qstars,
                   coefs_linear,
                   timestep=1.,
                   x0=0.5,
                   flux_scalings=None,
                   x=None,
                   dx=None,
                   fluxes=None,
                   dfluxes=None,
                   niter=None,
                   enforce_potential_inflow_pos=True,
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
    fluxes, dfluxes, x, dx, dQstars_ext : np.ndarray
        Arrays used by for computation "in place", i.e.
        when output arrays are not allocated within the
        function. This is useful if the function is used repeatedly.
        If None, the arrays are allocated during every function run.
    enforce_potential_inflow_pos: bool
        Verifies that potential inflows are always positive.
    errors: str
        - if 'ignore', then skip over error during time step integration
        - if 'raise', throw a ValueError.
        - if 'warn', throw a warning.

    Returns
    -------
    niter : np.ndarray
        Number of iterations (i.e. number of interpolation bands crossed).
        Vector of length P (one number for each timestep).
    x : np.ndarray
        Final value of x at the end of each timestep.
        Vector of length P.
    dx : np.ndarray
        Derivative of x against each independent variable at the end of each timestep.
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
    enforce_potential_inflow_pos = int(enforce_potential_inflow_pos)

    if flux_scalings is None:
        flux_scalings = np.ones(NFLUXES_INPUTS)

    if niter is None:
        niter = np.zeros(nval, dtype=np.int32)

    if x is None:
        x = np.zeros(nval, dtype=np.float64)

    if dx is None:
        dx = np.zeros((nval, nvariables), dtype=np.float64)

    if fluxes is None:
        fluxes = np.zeros((nval, NFLUXES_OUTPUTS), dtype=np.float64)

    if dfluxes is None:
        dfluxes = np.zeros((nval, nvariables, NFLUXES_OUTPUTS),
                           dtype=np.float64)

    ierrors = get_errorsid(errors)
    ierr = c_pyquasoare_grad.grad_reservoir(ierrors,
                                            enforce_potential_inflow_pos,
                                            alphas,
                                            timestep,
                                            gamma,
                                            flux_scalings,
                                            x0,
                                            Qstars,
                                            coefs_linear,
                                            niter,
                                            x,
                                            dx,
                                            fluxes,
                                            dfluxes)
    name = "c_pyquasoare_grad.grad_reservoir"
    error_management(errors, ierr, name)

    return niter, x, dx, fluxes, dfluxes


def grad_model(alphas, gammas,
               input_mapping,
               input_mapping_weights,
               x0,
               variables_reservoir_id,
               variables_type_id,
               Qstars,
               coefs_linear,
               timestep=1.,
               flux_scalings=None,
               x=None,
               dx=None,
               fluxes=None,
               dfluxes=None,
               niter=None,
               dQstars_ext=None,
               enforce_potential_inflow_pos=True,
               errors="ignore"):
    nreservoirs = alphas.shape[0]
    nval = Qstars.shape[0]
    enforce_potential_inflow_pos = int(enforce_potential_inflow_pos)

    if variables_reservoir_id is None:
        variables_reservoir_id = np.arange(nreservoirs, dtype=np.int32)
    nvariables = variables_reservoir_id.shape[0]

    if flux_scalings is None:
        flux_scalings = np.ones((nreservoirs, NFLUXES_INPUTS))

    if x is None:
        x = np.zeros((nval, nreservoirs), dtype=np.float64)

    if dx is None:
        dx = np.zeros((nval, nreservoirs, nvariables), dtype=np.float64)

    if fluxes is None:
        fluxes = np.zeros((nval, NFLUXES_OUTPUTS), dtype=np.float64)

    if dfluxes is None:
        dfluxes = np.zeros((nval, nvariables, NFLUXES_OUTPUTS),
                           dtype=np.float64)

    if dQstars_ext is None:
        dQstars_ext = np.zeros((nvariables, NFLUXES_INPUTS), dtype=np.float64)

    ierrors = get_errorsid(errors)
    ierr = c_pyquasoare_grad.grad_model(ierrors,
                                        enforce_potential_inflow_pos,
                                        alphas,
                                        timestep,
                                        gammas,
                                        input_mapping,
                                        input_mapping_weights,
                                        flux_scalings,
                                        x0,
                                        variables_reservoir_id,
                                        variables_type_id,
                                        Qstars,
                                        dQstars_ext,
                                        coefs_linear,
                                        x,
                                        dx,
                                        fluxes,
                                        dfluxes)
    name = "c_pyquasoare_grad.grad_model"
    error_management(errors, ierr, name)

    return x, fluxes, dfluxes


class NetworkLink():
    def __init__(self,
                 upstream_reservoir_id,
                 upstream_output_flux_id,
                 downstream_reservoir_id,
                 downstream_potential_flux_id,
                 weight, nres):
        if nres >= INPUT_MAPPING_MISSINGRESERVOIRID:
            errmsg = f"nreservoirs cannot exceed {INPUT_MAPPING_MISSINGRESERVOIRID}."
            raise ValueError(errmsg)
        self.nreservoirs = nres

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

        self.upstream_reservoir_id = upid
        self.upstream_output_flux_id = upoid
        self.downstream_reservoir_id = downid
        self.downstream_potential_flux_id = downiid
        self.weight = float(weight)

    def clone(self):
        return NetworkLink(self.upstream_reservoir_id,
                           self.upstream_output_flux_id,
                           self.downstream_reservoir_id,
                           self.downstream_potential_flux_id,
                           self.weight, self.nreservoirs)


class Network():
    def __init__(self, nreservoirs):
        if nreservoirs >= INPUT_MAPPING_MISSINGRESERVOIRID:
            errmsg = f"nreservoirs cannot exceed {INPUT_MAPPING_MISSINGRESERVOIRID}."
            raise ValueError(errmsg)

        self.nreservoirs = int(nreservoirs)
        self.links = []
        self._input_mapping = None
        self._input_mapping_weights = None

    def clone(self):
        netw = Network(self.nreservoirs)
        for lk in self.links:
            netw.links.append(lk.clone())
        return netw

    def get_input_mapping(self):
        nout = NFLUXES_OUTPUTS
        nres = self.nreservoirs
        nipmd = INPUT_MAPPING_LASTDIM

        ipm = self._input_mapping
        ipmw = self._input_mapping_weights
        if ipm is not None and ipmw is not None:
            return ipm, ipmw

        ipm = np.zeros((nres, nres * nout, nipmd), dtype=np.int32)
        ipm[:, :, 0] = INPUT_MAPPING_MISSINGRESERVOIRID

        ipmw = np.zeros((nres, nres * nout), dtype=np.float64)

        for lk in self.links:
            rdid = lk.downstream_reservoir_id
            idid = lk.downstream_potential_flux_id
            ruid = lk.upstream_reservoir_id
            ouid = lk.upstream_output_flux_id

            lks = ipm[rdid]
            ilk = np.where(lks[:, 0] == INPUT_MAPPING_MISSINGRESERVOIRID)[0][0]
            if ilk >= ipm.shape[1]:
                errmsg = "Exceeded the maximum number of inter-reservoir links"
                raise ValueError(errmsg)

            lks[ilk, :] = [ruid, ouid, idid]
            ipmw[rdid, ilk] = lk.weight

        self._input_mapping = ipm
        self._input_mapping_weights = ipmw

        return ipm, ipmw

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

        upids = set(add_upids(reservoir_id, [])[1:])
        if any(u >= reservoir_id for u in upids):
            errmsg = "Some reservoir ids located upstream of reservoir"\
                     + f" #{reservoir_id} are >={reservoir_id}."\
                     + " Please rearrange reservoirs to get lower ids upstream."
            raise ValueError(errmsg)
        return upids

    def add_link(self, upstream_reservoir_id, upstream_output_flux_id,
                 downstream_reservoir_id, downstream_potential_flux_id,
                 weight):
        nres = self.nreservoirs
        lk = NetworkLink(upstream_reservoir_id,
                         upstream_output_flux_id,
                         downstream_reservoir_id,
                         downstream_potential_flux_id,
                         weight, nres)
        self.links.append(lk)

        # check non-cyclic
        self.upstream_reservoirs(downstream_reservoir_id)


    def add_link_outflow(self, upstream_reservoir_id,
                         downstream_reservoir_id,
                         conversion_factor=1.):
        """ Links outflow from upstream to inflow for downstream.
        Weight is -1 because we need to convert the outflow (<0) from
        upstream reservoir to positive inflow to the downstream reservoir.
        """
        if conversion_factor <= 0:
            errmsg = "Expected conversion_factor > 0."
            raise ValueError(errmsg)

        self.add_link(upstream_reservoir_id, IFLUX_OUTFLOW,
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


class MultiReservoirModel():
    def __init__(self, nreservoirs,
                 nalphas=10,
                 errors="ignore",
                 enforce_potential_inflow_pos=True):
        self._enforce_potential_inflow_pos = enforce_potential_inflow_pos
        self._network = Network(nreservoirs)
        self._errors = errors
        self._gammas = None

        self._flux_scalings = np.ones((nreservoirs, NFLUXES_INPUTS))

        self.nalphas = nalphas
        self._alphas = get_alphas(nalphas,
                                  nreservoirs=nreservoirs)

        self._coefs_linear = None

        self.variables_reservoir_id = None
        self.variables_type_id = None
        self.Qstars = None
        self.dQstars_ext = None
        self.x0 = None
        self.x = None
        self.dx = None
        self.fluxes = None
        self.dfluxes = None
        self.niter = None
        self.timestep = None
        self.ntimesteps = None

    def check_reservoir_id(self, reservoir_id):
        if reservoir_id < 0 or reservoir_id >= self.nreservoirs:
            errmsg = f"Expected reservoir_id in [0, {self.nreservoirs}[."
            raise ValueError(errmsg)

    def check_allocated(self):
        if self.Qstars is None:
            errmsg = "Qstars is not allocated. Run 'allocate'."
            raise ValueError(errmsg)

    @property
    def errors(self):
        return self._errors

    @property
    def network(self):
        return self._network

    @property
    def nreservoirs(self):
        return self.network.nreservoirs

    def get_input_mapping(self):
        return self.network.get_input_mapping()

    @property
    def coefs_linear(self):
        return self._coefs_linear

    @property
    def alphas(self):
        al = self._alphas
        if al is None:
            raise ValueError("alphas are not set.")
        return al

    @property
    def gammas(self):
        gm = self._gammas
        if gm is None:
            raise ValueError("gammas are not set.")
        return gm

    @gammas.setter
    def gammas(self, value):
        nres = self.nreservoirs
        if isinstance(value, float) or isinstance(value, int):
            self._gammas = value * np.ones(nres)
        else:
            value = np.array(value)
            if value.shape != (nres, ):
                errmsg = f"Expected gammas of dim ({nres},)."\
                         + f" Got {value.shape}."
                raise ValueError(errmsg)

            self._gammas = value.astype(np.float64)

    @property
    def flux_scalings(self):
        fs = self._flux_scalings
        if fs is None:
            raise ValueError("flux_scalings are not set.")
        return fs

    @flux_scalings.setter
    def flux_scalings(self, value):
        nres = self.nreservoirs
        nfxi = NFLUXES_INPUTS
        if isinstance(value, float) or isinstance(value, int):
            self._flux_scalings = value * np.ones((nres, nfxi))
        else:
            value = np.array(value)
            if value.shape != (nres, nfxi):
                errmsg = f"Expected flux_scalings of dim ({nres},{nfxi})."\
                         + f" Got {value.shape}."
                raise ValueError(errmsg)

            self._flux_scalings = value.astype(np.float64)

    def add_link(self, upstream_reservoir_id, upstream_output_flux_id,
                 downstream_reservoir_id, downstream_potential_flux_id,
                 weight):
        self.network.add_link(upstream_reservoir_id, upstream_output_flux_id,
                              downstream_reservoir_id, downstream_potential_flux_id,
                              weight)

    def add_link_outflow(self, upstream_reservoir_id,
                         downstream_reservoir_id,
                         conversion_factor=1.):
        self.network.add_link_outflow(upstream_reservoir_id,
                                      downstream_reservoir_id,
                                      conversion_factor)

    def add_link_bypass_shortfall(self, upstream_reservoir_id,
                                  downstream_reservoir_id,
                                  conversion_factor=1.):
        self.network.add_link_bypass_shortfall(upstream_reservoir_id,
                                               downstream_reservoir_id,
                                               conversion_factor)

    def set_alphas(self, reservoir_id,
                   amin=0., amax=1., eps=1e-5):
        self.check_reservoir_id(reservoir_id)
        al = get_alphas(self.nalphas, amin, amax,
                        nreservoirs=1, eps=eps)
        self._alphas[reservoir_id] = al

    def set_coefs_linear(self, reservoir_id, flux_functions):
        self.check_reservoir_id(reservoir_id)

        sf = str(flux_functions)
        if sf in FLUX_FUNCTIONS:
            flux_functions = FLUX_FUNCTIONS[sf]

        nalphas = self.alphas.shape[1]
        nres = self.nreservoirs
        if self._coefs_linear is None:
            nfxi = NFLUXES_INPUTS
            nfxp = NFLUXES_POTENTIAL
            self._coefs_linear = np.zeros((nres, nfxi, nalphas - 1, nfxp))

        rid = reservoir_id
        ralphas = self.alphas[rid]
        self._coefs_linear[rid] = lin_coefficient_matrix(flux_functions, ralphas)

    def set_Qstars(self, reservoir_id, Qstars):
        self.check_allocated()
        if reservoir_id < 0 or reservoir_id >= self.nreservoirs:
            errmsg = f"Expected reservoir_id in [0, {self.nreservoirs}[."
            raise ValueError(errmsg)

        if Qstars.shape != (self.ntimesteps, NFLUXES_POTENTIAL):
            errmsg = "Expected Qstars.shape to be"\
                     + f" ({self.ntimesteps}, {NFLUXES_POTENTIAL}),"\
                     + f" got {Qstars.shape}."
            raise ValueError(errmsg)

        self.Qstars[:, reservoir_id, :] = Qstars

    def allocate(self, ntimesteps, variables_specs=None, timestep=1.):
        self.timestep = timestep
        self.ntimesteps = int(ntimesteps)

        nres = self.nreservoirs
        nfxi = NFLUXES_INPUTS
        nfxp = NFLUXES_POTENTIAL
        nfxo = NFLUXES_OUTPUTS

        if variables_specs is None:
            # no derivative variables, i.e. no gradient (pure simulation)
            self.variables_reservoir_id = np.zeros((0), dtype=np.int32)
            self.variables_type_id = np.zeros((0), dtype=np.int32)
            nvariables = 0
        else:
            variables_specs = np.array(variables_specs)
            self.variables_reservoir_id = \
                np.ascontiguousarray(variables_specs[:, 0]).astype(np.int32)
            self.variables_type_id = \
                np.ascontiguousarray(variables_specs[:, 1]).astype(np.int32)
            nvariables = len(variables_specs)

        Qst = np.zeros((ntimesteps, nres, nfxp))
        Qst.fill(np.nan)
        self.Qstars = Qst
        self.dQstars_ext = np.zeros((nvariables, nfxi))
        self.x0 = 0.5 * np.ones(nres)
        self.x = np.zeros((ntimesteps, nres))
        self.dx = np.zeros((ntimesteps, nres, nvariables))
        self.fluxes = np.zeros((ntimesteps, nres, nfxo))
        self.dfluxes = np.zeros((ntimesteps, nvariables, nres, nfxo))
        self.niter = np.zeros((ntimesteps, nres), dtype=np.int32)

    def clone(self):
        mcl = MultiReservoirModel(self.nreservoirs,
                                  self.nalphas,
                                  self.errors)
        mcl._network = self.network.clone()

        attrns = [
            "variables_reservoir_id",
            "variables_type_id",
            "_coefs_linear",
            "_flux_scalings",
            "_alphas",
            "_gammas",
            "Qstars",
            "x0",
            "x",
            "dx",
            "fluxes",
            "dfluxes",
            "niter",
            "timestep",
            "ntimesteps"
            ]
        for attrn in attrns:
            value = getattr(self, attrn)
            if hasattr(value, "copy"):
                value = value.copy()
            setattr(mcl, attrn, value)

        return mcl

    def run(self):
        self.check_allocated()

        # Get input mapping variables and weights
        inpm, inpmw = self.get_input_mapping()

        grad_model(alphas=self.alphas,
                   gammas=self.gammas,
                   input_mapping=inpm,
                   input_mapping_weights=inpmw,
                   x0=self.x0,
                   variables_reservoir_id=self.variables_reservoir_id,
                   variables_type_id=self.variables_type_id,
                   Qstars=self.Qstars,
                   coefs_linear=self.coefs_linear,
                   timestep=self.timestep,
                   flux_scalings=self.flux_scalings,
                   x=self.x,
                   dx=self.dx,
                   fluxes=self.fluxes,
                   dfluxes=self.dfluxes,
                   niter=self.niter,
                   dQstars_ext=self.dQstars_ext,
                   errors=self.errors,
                   enforce_potential_inflow_pos=self._enforce_potential_inflow_pos)

