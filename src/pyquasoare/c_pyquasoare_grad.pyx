import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_quasoare_grad.h':
    cdef int QUASOARE_GRAD_NPIECEWISEPARAMS
    cdef int QUASOARE_GRAD_NFLUXES_POTENTIAL
    cdef int QUASOARE_GRAD_NFLUXES_INPUTS
    cdef int QUASOARE_GRAD_NFLUXES_OUTPUTS
    cdef int QUASOARE_GRAD_NDERIV

    cdef int QUASOARE_GRAD_INPUT_MAPPING_LASTDIM
    cdef int QUASOARE_GRAD_INPUT_MAPPING_SCALINGFACTOR

    cdef int QUASOARE_GRAD_VARTYPE_NUMBER

    cdef int QUASOARE_GRAD_IFLUX_INFLOW
    cdef int QUASOARE_GRAD_IFLUX_DEMAND
    cdef int QUASOARE_GRAD_IFLUX_OUTFLOW
    cdef int QUASOARE_GRAD_IFLUX_BYPASS
    cdef int QUASOARE_GRAD_IFLUX_SHORTFALL
    cdef int QUASOARE_GRAD_IFLUX_TOTALOUT

    int c_grad_integrate(int nalphas, int nreservoirs, int nvariables,
                         double *alphas,
                         double timestep,
                         double gamma,
                         int *input_mapping,
                         double *Qstars,
                         double *dQstars_ext,
                         double *flux_scalings,
                         double *coefs_linear,
                         double t0,
                         double x0,
                         double *dx0,
                         int current_reservoir_id,
                         int *variables_reservoir_id,
                         int *variables_type_id,
                         int *niter,
                         double *x1,
                         double *dx1,
                         double *fluxes,
                         double *dfluxes);

    int c_grad_reservoir(int nalphas, int nval,
                         int errors,
                         double *alphas,
                         double timestep,
                         double gamma,
                         double *flux_scalings,
                         double x0,
                         double *dx0,
                         double *Qstars,
                         double *dQstars_ext,
                         double *coefs_linear,
                         int *niter,
                         double *x1,
                         double *dx1,
                         double *fluxes,
                         double *dfluxes);

    int c_grad_model(int nalphas, int nreservoirs, int nvariables, int nval,
                     int errors,
                     double *alphas,
                     double timestep,
                     double *gammas,
                     int *input_mapping,
                     double *flux_scalings,
                     double x0,
                     double *dx0,
                     int *variables_reservoir_id,
                     int *variables_type_id,
                     double *Qstars,
                     double *dQstars_ext,
                     double *coefs_linear,
                     double *x1,
                     double *dx1,
                     double *fluxes,
                     double *dfluxes);

# Flux sizes
NFLUXES_POTENTIAL = QUASOARE_GRAD_NFLUXES_POTENTIAL
NFLUXES_INPUTS = QUASOARE_GRAD_NFLUXES_INPUTS
NFLUXES_OUTPUTS = QUASOARE_GRAD_NFLUXES_OUTPUTS
NPIECEWISEPARAMS = QUASOARE_GRAD_NPIECEWISEPARAMS
NDERIV = QUASOARE_GRAD_NDERIV
INPUT_MAPPING_LASTDIM = QUASOARE_GRAD_INPUT_MAPPING_LASTDIM
INPUT_MAPPING_SCALINGFACTOR = QUASOARE_GRAD_INPUT_MAPPING_SCALINGFACTOR
VARTYPE_NUMBER = QUASOARE_GRAD_VARTYPE_NUMBER

IFLUX_INFLOW = QUASOARE_GRAD_IFLUX_INFLOW
IFLUX_DEMAND = QUASOARE_GRAD_IFLUX_DEMAND
IFLUX_OUTFLOW = QUASOARE_GRAD_IFLUX_OUTFLOW
IFLUX_BYPASS = QUASOARE_GRAD_IFLUX_BYPASS
IFLUX_SHORTFALL = QUASOARE_GRAD_IFLUX_SHORTFALL
IFLUX_TOTALOUT = QUASOARE_GRAD_IFLUX_TOTALOUT


def __cinit__(self):
    pass

cdef struct bidule:
    double * data1
    double * data2


def check_dim_input_mapping(input_mapping, nreservoirs):
    if input_mapping.shape[0] != nreservoirs:
        raise ValueError("input_mapping.shape[0] != nreservoirs")

    if input_mapping.shape[1] != nreservoirs * NFLUXES_OUTPUTS:
        raise ValueError("input_mapping.shape[1] != nreservoirs * NFLUXES_OUTPUTS")

    if input_mapping.shape[2] != INPUT_MAPPING_LASTDIM:
        raise ValueError("input_mapping.shape[2] != INPUT_MAPPING_LASTDIM")


def check_dim_coefs_linear(coefs_linear, nalphas, nreservoirs):
    ibase = 0 if nreservoirs == 1 else 1
    if nreservoirs != 1:
        if coefs_linear.shape[0] != nreservoirs:
            raise ValueError("coefs_linear.shape[0] != nreservoirs")

    if coefs_linear.shape[ibase] != NFLUXES_INPUTS:
        raise ValueError(f"coefs_linear.shape[{ibase}] != NFLUXES_INPUTS")

    if coefs_linear.shape[ibase + 1] != nalphas - 1:
        raise ValueError(f"coefs_linear.shape[{ibase + 1}] != nalphas - 1")

    if coefs_linear.shape[ibase + 2] != NFLUXES_POTENTIAL:
        raise ValueError(f"coefs_linear.shape[{ibase + 2}] != NFLUXES_POTENTIAL")


def grad_integrate(int nreservoirs, int current_reservoir_id,
                   np.ndarray[double, ndim=1, mode='c'] alphas not None,
                   double timestep,
                   double gamma,
                   np.ndarray[int, ndim=3, mode='c'] input_mapping not None,
                   np.ndarray[double, ndim=1, mode='c'] Qstars not None,
                   np.ndarray[double, ndim=2, mode='c'] dQstars_ext not None,
                   np.ndarray[double, ndim=1, mode='c'] flux_scalings not None,
                   np.ndarray[double, ndim=3, mode='c'] coefs_linear not None,
                   double t0, double x0,
                   np.ndarray[double, ndim=1, mode='c'] dx0 not None,
                   np.ndarray[int, ndim=1, mode='c'] variables_reservoir_id not None,
                   np.ndarray[int, ndim=1, mode='c'] variables_type_id not None,
                   np.ndarray[int, ndim=1, mode='c'] niter not None,
                   np.ndarray[double, ndim=1, mode='c'] x1 not None,
                   np.ndarray[double, ndim=1, mode='c'] dx1 not None,
                   np.ndarray[double, ndim=1, mode='c'] fluxes not None,
                   np.ndarray[double, ndim=3, mode='c'] dfluxes not None):

    # Check dimensions
    cdef int nalphas = alphas.shape[0]
    cdef int nvariables = dx0.shape[0]

    check_dim_input_mapping(input_mapping, nreservoirs)
    check_dim_coefs_linear(coefs_linear, nalphas, 1)

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

    if x1.shape[0] != 1:
        raise ValueError("s1.shape[0] != 1")

    if dx1.shape[0] != nvariables:
        raise ValueError("dx1.shape[0] != nvariables")

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
                           <double*> np.PyArray_DATA(alphas),
                           timestep,
                           gamma,
                           <int*> np.PyArray_DATA(input_mapping),
                           <double*> np.PyArray_DATA(Qstars),
                           <double*> np.PyArray_DATA(dQstars_ext),
                           <double*> np.PyArray_DATA(flux_scalings),
                           <double*> np.PyArray_DATA(coefs_linear),
                           t0, x0,
                           <double*> np.PyArray_DATA(dx0),
                           current_reservoir_id,
                           <int*> np.PyArray_DATA(variables_reservoir_id),
                           <int*> np.PyArray_DATA(variables_type_id),
                           <int*> np.PyArray_DATA(niter),
                           <double*> np.PyArray_DATA(x1),
                           <double*> np.PyArray_DATA(dx1),
                           <double*> np.PyArray_DATA(fluxes),
                           <double*> np.PyArray_DATA(dfluxes))

def grad_reservoir(int errors,
                   np.ndarray[double, ndim=1, mode='c'] alphas not None,
                   double timestep,
                   double gamma,
                   np.ndarray[double, ndim=1, mode='c'] flux_scalings not None,
                   double x0,
                   np.ndarray[double, ndim=1, mode='c'] dx0 not None,
                   np.ndarray[double, ndim=2, mode='c'] Qstars not None,
                   np.ndarray[double, ndim=2, mode='c'] dQstars_ext not None,
                   np.ndarray[double, ndim=3, mode='c'] coefs_linear not None,
                   np.ndarray[int, ndim=1, mode='c'] niter not None,
                   np.ndarray[double, ndim=1, mode='c'] x1 not None,
                   np.ndarray[double, ndim=2, mode='c'] dx1 not None,
                   np.ndarray[double, ndim=2, mode='c'] fluxes not None,
                   np.ndarray[double, ndim=3, mode='c'] dfluxes not None):

    # Check dimensions
    cdef int nalphas = alphas.shape[0]
    cdef int nval = Qstars.shape[0]
    cdef int nvariables = VARTYPE_NUMBER
    cdef int nreservoirs = 1

    check_dim_coefs_linear(coefs_linear, nalphas, nreservoirs)

    if flux_scalings.shape[0] != NFLUXES_INPUTS:
        raise ValueError("flux_scalings.shape[0] != NFLUXES_INPUTS")

    if Qstars.shape[1] != NFLUXES_POTENTIAL:
        raise ValueError("Qstars.shape[1] != NFLUXES_POTENTIAL")

    if dQstars_ext.shape[0] != nvariables:
        raise ValueError("dQstars_ext.shape[0] != nvariables")

    if dQstars_ext.shape[1] != NFLUXES_INPUTS:
        raise ValueError("dQstars_ext.shape[1] != NFLUXES_INPUTS")

    if x1.shape[0] != nval:
        raise ValueError("x1.shape[0] != nval")

    if dx1.shape[0] != nval:
        raise ValueError("dx1.shape[0] != nval")

    if dx1.shape[1] != nvariables:
        raise ValueError("dx1.shape[1] != nvariables")

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
                            <double*> np.PyArray_DATA(alphas),
                            timestep,
                            gamma,
                            <double*> np.PyArray_DATA(flux_scalings),
                            x0,
                            <double*> np.PyArray_DATA(dx0),
                            <double*> np.PyArray_DATA(Qstars),
                            <double*> np.PyArray_DATA(dQstars_ext),
                            <double*> np.PyArray_DATA(coefs_linear),
                            <int*> np.PyArray_DATA(niter),
                            <double*> np.PyArray_DATA(x1),
                            <double*> np.PyArray_DATA(dx1),
                            <double*> np.PyArray_DATA(fluxes),
                            <double*> np.PyArray_DATA(dfluxes))


def grad_model(int errors,
               np.ndarray[double, ndim=2, mode='c'] alphas not None,
               double timestep,
               np.ndarray[double, ndim=1, mode='c'] gammas not None,
               np.ndarray[int, ndim=3, mode='c'] input_mapping not None,
               np.ndarray[double, ndim=2, mode='c'] flux_scalings not None,
               double x0,
               np.ndarray[double, ndim=1, mode='c'] dx0 not None,
               np.ndarray[int, ndim=1, mode='c'] variables_reservoir_id not None,
               np.ndarray[int, ndim=1, mode='c'] variables_type_id not None,
               np.ndarray[double, ndim=2, mode='c'] Qstars not None,
               np.ndarray[double, ndim=2, mode='c'] dQstars_ext not None,
               np.ndarray[double, ndim=4, mode='c'] coefs_linear not None,
               np.ndarray[double, ndim=2, mode='c'] x1 not None,
               np.ndarray[double, ndim=3, mode='c'] dx1 not None,
               np.ndarray[double, ndim=2, mode='c'] fluxes not None,
               np.ndarray[double, ndim=4, mode='c'] dfluxes not None):

    # Check dimensions
    cdef int nreservoirs = alphas.shape[0]
    cdef int nalphas = alphas.shape[1]
    cdef int nval = Qstars.shape[0]
    cdef int nvariables = dx0.shape[0]

    check_dim_input_mapping(input_mapping, nreservoirs)
    check_dim_coefs_linear(coefs_linear, nalphas, nreservoirs)

    if gammas.shape[0] != nreservoirs:
        raise ValueError("gammas.shape[0] != nreservoirs")

    if flux_scalings.shape[0] != NFLUXES_INPUTS * nreservoirs:
        raise ValueError("flux_scalings.shape[0] != NFLUXES_INPUTS x nreservoirs")

    if variables_reservoir_id.shape[0] != nvariables:
        raise ValueError("variables_reservoir_id.shape[0] != nvariables")

    if variables_type_id.shape[0] != nvariables:
        raise ValueError("variables_type_id.shape[0] != nvariables")

    if Qstars.shape[1] != NFLUXES_POTENTIAL * nreservoirs:
        raise ValueError("Qstars.shape[1] != NFLUXES_POTENTIAL * nreservoirs")

    if dQstars_ext.shape[0] != nvariables:
        raise ValueError("dQstars_ext.shape[0] != nvariables")

    if dQstars_ext.shape[1] != NFLUXES_INPUTS:
        raise ValueError("dQstars_ext.shape[1] != NFLUXES_INPUTS")

    if x1.shape[0] != nval:
        raise ValueError("x1.shape[0] != nval")

    if x1.shape[1] != nreservoirs:
        raise ValueError("x1.shape[1] != nreservoirs")

    if dx1.shape[0] != nval:
        raise ValueError("dx1.shape[0] != nval")

    if dx1.shape[1] != nreservoirs:
        raise ValueError("dx1.shape[1] != nreservoirs")

    if dx1.shape[2] != nvariables:
        raise ValueError("dx1.shape[2] != nvariables")

    if fluxes.shape[0] != nval:
        raise ValueError("fluxes.shape[0] != nval")

    if fluxes.shape[1] != NFLUXES_OUTPUTS * nreservoirs:
        raise ValueError("fluxes.shape[0] != NFLUXES_OUTPUTS * nreservoirs")

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
                        <double*> np.PyArray_DATA(alphas),
                        timestep,
                        <double*> np.PyArray_DATA(gammas),
                        <int*> np.PyArray_DATA(input_mapping),
                        <double*> np.PyArray_DATA(flux_scalings),
                        x0,
                        <double*> np.PyArray_DATA(dx0),
                        <int*> np.PyArray_DATA(variables_reservoir_id),
                        <int*> np.PyArray_DATA(variables_type_id),
                        <double*> np.PyArray_DATA(Qstars),
                        <double*> np.PyArray_DATA(dQstars_ext),
                        <double*> np.PyArray_DATA(coefs_linear),
                        <double*> np.PyArray_DATA(x1),
                        <double*> np.PyArray_DATA(dx1),
                        <double*> np.PyArray_DATA(fluxes),
                        <double*> np.PyArray_DATA(dfluxes))
