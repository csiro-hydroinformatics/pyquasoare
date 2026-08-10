import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_quasoare_grad.h':
    cdef unsigned int QUASOARE_GRAD_NPIECEWISEPARAMS
    cdef unsigned int QUASOARE_GRAD_NFLUXES_POTENTIAL
    cdef unsigned int QUASOARE_GRAD_NFLUXES_INPUTS
    cdef unsigned int QUASOARE_GRAD_NFLUXES_OUTPUTS
    cdef unsigned int QUASOARE_GRAD_NDERIV

    cdef unsigned int QUASOARE_GRAD_INPUT_MAPPING_LASTDIM
    cdef unsigned int QUASOARE_GRAD_INPUT_MAPPING_MISSINGRESERVOIRID

    cdef unsigned int QUASOARE_GRAD_VARTYPE_NUMBER

    cdef unsigned int QUASOARE_GRAD_IFLUX_INFLOW
    cdef unsigned int QUASOARE_GRAD_IFLUX_DEMAND
    cdef unsigned int QUASOARE_GRAD_IFLUX_DRAINAGE
    cdef unsigned int QUASOARE_GRAD_IFLUX_BYPASS
    cdef unsigned int QUASOARE_GRAD_IFLUX_SHORTFALL
    cdef unsigned int QUASOARE_GRAD_IFLUX_OUTFLOW

    unsigned int c_grad_integrate(int nalphas, unsigned int nreservoirs, unsigned int nvariables,
                         unsigned int check_potential_inflow_pos,
                         double *alphas,
                         double timestep,
                         double gamma,
                         unsigned int *input_mapping,
                         double *input_mapping_weights,
                         double *Qstars,
                         double *dQstars_ext,
                         double *flux_scalings,
                         double *coefs_linear,
                         unsigned int current_reservoir_id,
                         unsigned int *variables_reservoir_id,
                         unsigned int *variables_type_id,
                         unsigned int *niter,
                         double *x,
                         double *dx,
                         double *fluxes,
                         double *dfluxes);

    unsigned int c_grad_reservoir(int nalphas, unsigned int nval,
                         unsigned int errors,
                         unsigned int check_potential_inflow_pos,
                         double *alphas,
                         double timestep,
                         double gamma,
                         double *flux_scalings,
                         double x0,
                         double *Qstars,
                         double *coefs_linear,
                         unsigned int *niter,
                         double *x,
                         double *dx,
                         double *fluxes,
                         double *dfluxes);

    unsigned int c_grad_model(int nalphas, unsigned int nreservoirs, unsigned int nvariables, unsigned int nval,
                     unsigned int errors,
                     unsigned int check_potential_inflow_pos,
                     double *alphas,
                     double timestep,
                     double *gammas,
                     unsigned int *input_mapping,
                     double *input_mapping_weights,
                     double *flux_scalings,
                     double *x0,
                     unsigned int *variables_reservoir_id,
                     unsigned int *variables_type_id,
                     double *Qstars,
                     double *dQstars_ext,
                     double *coefs_linear,
                     double *x,
                     double *dx,
                     double *fluxes,
                     double *dfluxes);

# Flux sizes
NFLUXES_POTENTIAL = QUASOARE_GRAD_NFLUXES_POTENTIAL
NFLUXES_INPUTS = QUASOARE_GRAD_NFLUXES_INPUTS
NFLUXES_OUTPUTS = QUASOARE_GRAD_NFLUXES_OUTPUTS
NPIECEWISEPARAMS = QUASOARE_GRAD_NPIECEWISEPARAMS
NDERIV = QUASOARE_GRAD_NDERIV
INPUT_MAPPING_LASTDIM = QUASOARE_GRAD_INPUT_MAPPING_LASTDIM
INPUT_MAPPING_MISSINGRESERVOIRID = QUASOARE_GRAD_INPUT_MAPPING_MISSINGRESERVOIRID
VARTYPE_NUMBER = QUASOARE_GRAD_VARTYPE_NUMBER

IFLUX_INFLOW = QUASOARE_GRAD_IFLUX_INFLOW
IFLUX_DEMAND = QUASOARE_GRAD_IFLUX_DEMAND
IFLUX_DRAINAGE = QUASOARE_GRAD_IFLUX_DRAINAGE
IFLUX_BYPASS = QUASOARE_GRAD_IFLUX_BYPASS
IFLUX_SHORTFALL = QUASOARE_GRAD_IFLUX_SHORTFALL
IFLUX_OUTFLOW = QUASOARE_GRAD_IFLUX_OUTFLOW


def __cinit__(self):
    pass


def check_dim_input_mapping(input_mapping, input_mapping_weights,
                            nreservoirs):
    if input_mapping.shape[0] != nreservoirs:
        raise ValueError("input_mapping.shape[0] != nreservoirs")

    if input_mapping.shape[1] != nreservoirs * NFLUXES_OUTPUTS:
        raise ValueError("input_mapping.shape[1] != nreservoirs * NFLUXES_OUTPUTS")

    if input_mapping.shape[2] != INPUT_MAPPING_LASTDIM:
        raise ValueError("input_mapping.shape[2] != INPUT_MAPPING_LASTDIM")

    if input_mapping_weights.shape[0] != nreservoirs:
        raise ValueError("input_mapping_weights.shape[0] != nreservoirs")

    if input_mapping_weights.shape[1] != nreservoirs * NFLUXES_OUTPUTS:
        raise ValueError("input_mapping_weights.shape[1] != nreservoirs * NFLUXES_OUTPUTS")

def check_dim_coefs_linear(coefs_linear, nalphas, nreservoirs, is_model):
    ibase = 1 if is_model  else 0
    if nreservoirs != 1:
        if coefs_linear.shape[0] != nreservoirs:
            raise ValueError("coefs_linear.shape[0] != nreservoirs")

    if coefs_linear.shape[ibase] != NFLUXES_INPUTS:
        raise ValueError(f"coefs_linear.shape[{ibase}] != NFLUXES_INPUTS")

    if coefs_linear.shape[ibase + 1] != nalphas - 1:
        raise ValueError(f"coefs_linear.shape[{ibase + 1}] != nalphas - 1")

    if coefs_linear.shape[ibase + 2] != NFLUXES_POTENTIAL:
        raise ValueError(f"coefs_linear.shape[{ibase + 2}] != NFLUXES_POTENTIAL")


def grad_integrate(int nreservoirs, unsigned int current_reservoir_id,
                   unsigned int check_potential_inflow_pos,
                   np.ndarray[double, ndim=1, mode='c'] alphas not None,
                   double timestep,
                   double gamma,
                   np.ndarray[int, ndim=3, mode='c'] input_mapping not None,
                   np.ndarray[double, ndim=2, mode='c'] input_mapping_weights not None,
                   np.ndarray[double, ndim=1, mode='c'] Qstars not None,
                   np.ndarray[double, ndim=2, mode='c'] dQstars_ext not None,
                   np.ndarray[double, ndim=1, mode='c'] flux_scalings not None,
                   np.ndarray[double, ndim=3, mode='c'] coefs_linear not None,
                   np.ndarray[int, ndim=1, mode='c'] variables_reservoir_id not None,
                   np.ndarray[int, ndim=1, mode='c'] variables_type_id not None,
                   np.ndarray[int, ndim=1, mode='c'] niter not None,
                   np.ndarray[double, ndim=1, mode='c'] x not None,
                   np.ndarray[double, ndim=1, mode='c'] dx not None,
                   np.ndarray[double, ndim=1, mode='c'] fluxes not None,
                   np.ndarray[double, ndim=3, mode='c'] dfluxes not None):

    # Check dimensions
    cdef unsigned int nalphas = alphas.shape[0]
    cdef unsigned int nvariables = dx.shape[0]

    check_dim_input_mapping(input_mapping, input_mapping_weights,
                            nreservoirs)
    check_dim_coefs_linear(coefs_linear, nalphas, 1, False)

    if Qstars.shape[0] != NFLUXES_POTENTIAL:
        raise ValueError("Qstars.shape[0] != NFLUXES_POTENTIAL")

    if dQstars_ext.shape[0] != nvariables:
        raise ValueError("dQstars_ext.shape[0] != nvariables")

    if dQstars_ext.shape[1] != NFLUXES_INPUTS:
        raise ValueError("dQstars_ext.shape[1] != NFLUXES_INPUTS")

    if flux_scalings.shape[0] != NFLUXES_INPUTS:
        raise ValueError("coefs_linear.shape[0] != NFLUXES_INPUTS")

    if variables_reservoir_id.shape[0] != nvariables:
        raise ValueError("variables_reservoir_id.shape[0] != nvariables")

    if variables_type_id.shape[0] != nvariables:
        raise ValueError("variables_type_id.shape[0] != nvariables")

    if niter.shape[0] != 1:
        raise ValueError("niter.shape[0] != 1")

    if x.shape[0] != 1:
        raise ValueError("x.shape[0] != 1")

    if fluxes.shape[0] != NFLUXES_OUTPUTS:
        raise ValueError("fluxes.shape[0] != NFLUXES_OUTPUTS")

    if dfluxes.shape[0] != nvariables:
        raise ValueError("dfluxes.shape[0] != nvariables")

    if dfluxes.shape[1] != nreservoirs:
        raise ValueError("dfluxes.shape[1] != nreservoirs")

    if dfluxes.shape[2] != NFLUXES_OUTPUTS:
        raise ValueError("dfluxes.shape[2] != NFLUXES_OUTPUTS")

    # Run C code
    return c_grad_integrate(nalphas, nreservoirs, nvariables,
                           check_potential_inflow_pos,
                           <double*> np.PyArray_DATA(alphas),
                           timestep,
                           gamma,
                           <unsigned int*> np.PyArray_DATA(input_mapping),
                           <double*> np.PyArray_DATA(input_mapping_weights),
                           <double*> np.PyArray_DATA(Qstars),
                           <double*> np.PyArray_DATA(dQstars_ext),
                           <double*> np.PyArray_DATA(flux_scalings),
                           <double*> np.PyArray_DATA(coefs_linear),
                           current_reservoir_id,
                           <unsigned int*> np.PyArray_DATA(variables_reservoir_id),
                           <unsigned int*> np.PyArray_DATA(variables_type_id),
                           <unsigned int*> np.PyArray_DATA(niter),
                           <double*> np.PyArray_DATA(x),
                           <double*> np.PyArray_DATA(dx),
                           <double*> np.PyArray_DATA(fluxes),
                           <double*> np.PyArray_DATA(dfluxes))

def grad_reservoir(int errors,
                   unsigned int check_potential_inflow_pos,
                   np.ndarray[double, ndim=1, mode='c'] alphas not None,
                   double timestep,
                   double gamma,
                   np.ndarray[double, ndim=1, mode='c'] flux_scalings not None,
                   double x0,
                   np.ndarray[double, ndim=2, mode='c'] Qstars not None,
                   np.ndarray[double, ndim=3, mode='c'] coefs_linear not None,
                   np.ndarray[int, ndim=1, mode='c'] niter not None,
                   np.ndarray[double, ndim=1, mode='c'] x not None,
                   np.ndarray[double, ndim=2, mode='c'] dx not None,
                   np.ndarray[double, ndim=2, mode='c'] fluxes not None,
                   np.ndarray[double, ndim=3, mode='c'] dfluxes not None):

    # Check dimensions
    cdef unsigned int nalphas = alphas.shape[0]
    cdef unsigned int nval = Qstars.shape[0]
    cdef unsigned int nvariables = VARTYPE_NUMBER
    cdef unsigned int nreservoirs = 1

    check_dim_coefs_linear(coefs_linear, nalphas, nreservoirs, False)

    if flux_scalings.shape[0] != NFLUXES_INPUTS:
        raise ValueError("flux_scalings.shape[0] != NFLUXES_INPUTS")

    if Qstars.shape[1] != NFLUXES_POTENTIAL:
        raise ValueError("Qstars.shape[1] != NFLUXES_POTENTIAL")

    if x.shape[0] != nval:
        raise ValueError("x.shape[0] != nval")

    if dx.shape[0] != nval:
        raise ValueError("dx.shape[0] != nval")

    if dx.shape[1] != nvariables:
        raise ValueError("dx.shape[1] != nvariables")

    if niter.shape[0] != nval:
        raise ValueError("niter.shape[0] != nval")

    if fluxes.shape[0] != nval:
        raise ValueError("fluxes.shape[0] != nval")

    if fluxes.shape[1] != NFLUXES_OUTPUTS:
        raise ValueError("fluxes.shape[1] != NFLUXES_OUTPUTS")

    if dfluxes.shape[0] != nval:
        raise ValueError("dfluxes.shape[0] != nval")

    if dfluxes.shape[1] != nvariables:
        raise ValueError("dfluxes.shape[1] != nvariables")

    if dfluxes.shape[2] != NFLUXES_OUTPUTS:
        raise ValueError("dfluxes.shape[2] != NFLUXES_OUTPUTS")

    # Run C code
    return c_grad_reservoir(nalphas, nval, errors,
                            check_potential_inflow_pos,
                            <double*> np.PyArray_DATA(alphas),
                            timestep,
                            gamma,
                            <double*> np.PyArray_DATA(flux_scalings),
                            x0,
                            <double*> np.PyArray_DATA(Qstars),
                            <double*> np.PyArray_DATA(coefs_linear),
                            <unsigned int*> np.PyArray_DATA(niter),
                            <double*> np.PyArray_DATA(x),
                            <double*> np.PyArray_DATA(dx),
                            <double*> np.PyArray_DATA(fluxes),
                            <double*> np.PyArray_DATA(dfluxes))


def grad_model(int errors,
               unsigned int check_potential_inflow_pos,
               np.ndarray[double, ndim=2, mode='c'] alphas not None,
               double timestep,
               np.ndarray[double, ndim=1, mode='c'] gammas not None,
               np.ndarray[int, ndim=3, mode='c'] input_mapping not None,
               np.ndarray[double, ndim=2, mode='c'] input_mapping_weights not None,
               np.ndarray[double, ndim=2, mode='c'] flux_scalings not None,
               np.ndarray[double, ndim=1, mode='c'] x0 not None,
               np.ndarray[int, ndim=1, mode='c'] variables_reservoir_id not None,
               np.ndarray[int, ndim=1, mode='c'] variables_type_id not None,
               np.ndarray[double, ndim=3, mode='c'] Qstars not None,
               np.ndarray[double, ndim=2, mode='c'] dQstars_ext not None,
               np.ndarray[double, ndim=4, mode='c'] coefs_linear not None,
               np.ndarray[double, ndim=2, mode='c'] x not None,
               np.ndarray[double, ndim=3, mode='c'] dx not None,
               np.ndarray[double, ndim=3, mode='c'] fluxes not None,
               np.ndarray[double, ndim=4, mode='c'] dfluxes not None):

    # Check dimensions
    cdef unsigned int nreservoirs = alphas.shape[0]
    cdef unsigned int nalphas = alphas.shape[1]
    cdef unsigned int nval = Qstars.shape[0]
    cdef unsigned int nvariables = dQstars_ext.shape[0]

    check_dim_input_mapping(input_mapping, input_mapping_weights,
                            nreservoirs)
    check_dim_coefs_linear(coefs_linear, nalphas, nreservoirs, True)

    if gammas.shape[0] != nreservoirs:
        raise ValueError("gammas.shape[0] != nreservoirs")

    if x0.shape[0] != nreservoirs:
        raise ValueError("x0.shape[0] != nreservoirs")

    if flux_scalings.shape[0] != nreservoirs:
        raise ValueError("flux_scalings.shape[0] != nreservoirs")

    if flux_scalings.shape[1] != NFLUXES_INPUTS:
        raise ValueError("flux_scalings.shape[1] != NFLUXES_INPUTS")

    if variables_reservoir_id.shape[0] != nvariables:
        raise ValueError("variables_reservoir_id.shape[0] != nvariables")

    if variables_type_id.shape[0] != nvariables:
        raise ValueError("variables_type_id.shape[0] != nvariables")

    if Qstars.shape[1] != nreservoirs:
        raise ValueError("Qstars.shape[1] != nreservoirs")

    if Qstars.shape[2] != NFLUXES_POTENTIAL:
        raise ValueError("Qstars.shape[2] != NFLUXES_POTENTIAL")

    if dQstars_ext.shape[1] != NFLUXES_INPUTS:
        raise ValueError("dQstars_ext.shape[1] != NFLUXES_INPUTS")

    if x.shape[0] != nval:
        raise ValueError("x.shape[0] != nval")

    if x.shape[1] != nreservoirs:
        raise ValueError("x.shape[1] != nreservoirs")

    if dx.shape[0] != nval:
        raise ValueError("dx.shape[0] != nval")

    if dx.shape[1] != nreservoirs:
        raise ValueError("dx.shape[1] != nreservoirs")

    if dx.shape[2] != nvariables:
        raise ValueError("dx.shape[2] != nvariables")

    if fluxes.shape[0] != nval:
        raise ValueError("fluxes.shape[0] != nval")

    if fluxes.shape[1] != nreservoirs:
        raise ValueError("fluxes.shape[0] != nreservoirs")

    if fluxes.shape[2] != NFLUXES_OUTPUTS:
        raise ValueError("fluxes.shape[1] != NFLUXES_OUTPUTS")

    if dfluxes.shape[0] != nval:
        raise ValueError("dfluxes.shape[0] != nval")

    if dfluxes.shape[1] != nvariables:
        raise ValueError("dfluxes.shape[1] != nvariables")

    if dfluxes.shape[2] != nreservoirs:
        raise ValueError("dfluxes.shape[2] != nreservoirs")

    if dfluxes.shape[3] != NFLUXES_OUTPUTS:
        raise ValueError("dfluxes.shape[3] != NFLUXES_OUTPUTS")

    # Run C code
    return c_grad_model(nalphas, nreservoirs, nvariables, nval, errors,
                        check_potential_inflow_pos,
                        <double*> np.PyArray_DATA(alphas),
                        timestep,
                        <double*> np.PyArray_DATA(gammas),
                        <unsigned int*> np.PyArray_DATA(input_mapping),
                        <double*> np.PyArray_DATA(input_mapping_weights),
                        <double*> np.PyArray_DATA(flux_scalings),
                        <double*> np.PyArray_DATA(x0),
                        <unsigned int*> np.PyArray_DATA(variables_reservoir_id),
                        <unsigned int*> np.PyArray_DATA(variables_type_id),
                        <double*> np.PyArray_DATA(Qstars),
                        <double*> np.PyArray_DATA(dQstars_ext),
                        <double*> np.PyArray_DATA(coefs_linear),
                        <double*> np.PyArray_DATA(x),
                        <double*> np.PyArray_DATA(dx),
                        <double*> np.PyArray_DATA(fluxes),
                        <double*> np.PyArray_DATA(dfluxes))
