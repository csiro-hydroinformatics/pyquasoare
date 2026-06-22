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

    int c_grad_integrate(int nalphas, int nreservoirs, int nvariables,
                         double * alphas,
                         double timestep,
                         double gamma,
                         double *input_mapping,
                         double *Qstars,
                         double *flux_scalings,
                         double * coefs_linear,
                         double t0,
                         double x0,
                         double *dx0,
                         int current_reservoir_id,
                         int *variables_reservoir_id,
                         int *variables_type_id,
                         int *niter,
                         double * x1,
                         double * dx1,
                         double *fluxes,
                         double * dfluxes);

    int c_grad_reservoir(int nalphas, int nreservoirs, int nvariables, int nval,
                        int errors,
                        double * alphas,
                        double timestep,
                        double gamma,
                        double * input_mapping,
                        double *flux_scalings,
                        double x0,
                        double * dx0,
                        int current_reservoir_id,
                        int *variables_reservoir_id,
                        int *variables_type_id,
                        double * Qstars,
                        double * coefs_linear,
                        int * niter,
                        double * x1,
                        double * dx1,
                        double * fluxes,
                        double * dfluxes);

    int c_grad_model(int nalphas, int nreservoirs, int nvariables, int nval,
                     int errors,
                     double * alphas,
                     double timestep,
                     double * gammas,
                     double * input_mapping,
                     double * flux_scalings,
                     double x0,
                     double * dx0,
                     int *variables_reservoir_id,
                     int *variables_type_id,
                     double * Qstars,
                     double * coefs_linear,
                     double * x1,
                     double * dx1,
                     double * fluxes,
                     double * dfluxes);

# Flux sizes
NFLUXES_POTENTIAL = QUASOARE_GRAD_NFLUXES_POTENTIAL
NFLUXES_INPUTS = QUASOARE_GRAD_NFLUXES_INPUTS
NFLUXES_OUTPUTS = QUASOARE_GRAD_NFLUXES_OUTPUTS
NPIECEWISEPARAMS = QUASOARE_GRAD_NPIECEWISEPARAMS
NDERIV = QUASOARE_GRAD_NDERIV


def __cinit__(self):
    pass


def grad_integrate(int nreservoirs, int current_reservoir_id,
                   np.ndarray[double, ndim=1, mode='c'] alphas not None,
                   double timestep,
                   double gamma,
                   np.ndarray[double, ndim=2, mode='c'] input_mapping not None,
                   np.ndarray[double, ndim=1, mode='c'] qstars not None,
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
                   np.ndarray[double, ndim=1, mode='c'] dfluxes not None):

    # Check dimensions
    cdef int nalphas = alphas.shape[0]
    cdef int nvariables = dx0.shape[0]

    if qstars.shape[0] != NFLUXES_POTENTIAL:
        raise ValueError("qstars.shape[0] != NFLUXES_POTENTIAL")

    if flux_scalings.shape[0] != NFLUXES_INPUTS:
        raise ValueError("coefs_linear.shape[0] != NFLUXES_INPUTS")

    if coefs_linear.shape[0] != NFLUXES_INPUTS:
        raise ValueError("coefs_linear.shape[0] != NFLUXES_INPUTS")

    if coefs_linear.shape[1] != nalphas - 1:
        raise ValueError("coefs_linear.shape[1] != nalphas - 1")

    if coefs_linear.shape[2] != NFLUXES_POTENTIAL:
        raise ValueError("coefs_linear.shape[2] != NFLUXES_POTENTIAL")

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

    # Run C code
    return c_grad_integrate(nalphas, nreservoirs, nvariables,
                           <double*> np.PyArray_DATA(alphas),
                           timestep,
                           gamma,
                           <double*> np.PyArray_DATA(input_mapping),
                           <double*> np.PyArray_DATA(qstars),
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
                   np.ndarray[double, ndim=2, mode='c'] input_mapping not None,
                   np.ndarray[double, ndim=1, mode='c'] flux_scalings not None,
                   double x0,
                   np.ndarray[double, ndim=1, mode='c'] dx0 not None,
                   int current_reservoir_id,
                   np.ndarray[int, ndim=1, mode='c'] variables_reservoir_id not None,
                   np.ndarray[int, ndim=1, mode='c'] variables_type_id not None,
                   np.ndarray[double, ndim=2, mode='c'] qstars not None,
                   np.ndarray[double, ndim=3, mode='c'] coefs_linear not None,
                   np.ndarray[int, ndim=1, mode='c'] niter not None,
                   np.ndarray[double, ndim=1, mode='c'] x1 not None,
                   np.ndarray[double, ndim=1, mode='c'] dx1 not None,
                   np.ndarray[double, ndim=2, mode='c'] fluxes not None,
                   np.ndarray[double, ndim=3, mode='c'] dfluxes not None):

    # Check dimensions
    cdef int nalphas = alphas.shape[0]
    cdef int nval = qstars.shape[0]
    cdef int nvariables = dx0.shape[0]
    cdef int nreservoirs = 1

    if flux_scalings.shape[0] != NFLUXES_INPUTS:
        raise ValueError("flux_scalings.shape[0] != NFLUXES_INPUTS")

    if variables_reservoir_id.shape[0] != nvariables:
        raise ValueError("variables_reservoir_id.shape[0] != nvariables")

    if any(variables_reservoir_id != 0):
        raise ValueError("variables_reservoir_id != 0")

    if variables_type_id.shape[0] != nvariables:
        raise ValueError("variables_type_id.shape[0] != nvariables")

    if qstars.shape[1] != NFLUXES_POTENTIAL:
        raise ValueError("qstars.shape[1] != NFLUXES_POTENTIAL")

    if coefs_linear.shape[0] != NFLUXES_INPUTS:
        raise ValueError("coefs_linear.shape[0] != NFLUXES_INPUTS")

    if coefs_linear.shape[1] != nalphas - 1:
        raise ValueError("coefs_linear.shape[1] != nalphas - 1")

    if coefs_linear.shape[2] != NFLUXES_POTENTIAL:
        raise ValueError("coefs_linear.shape[2] != NFLUXES_POTENTIAL")

    if x1.shape[0] != nval:
        raise ValueError("x1.shape[0] != nval")

    if dx1.shape[0] != nval:
        raise ValueError("dx1.shape[0] != nval")

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
    return c_grad_reservoir(nalphas, nreservoirs, nvariables, nval, errors,
                            <double*> np.PyArray_DATA(alphas),
                            timestep,
                            gamma,
                            <double*> np.PyArray_DATA(input_mapping),
                            <double*> np.PyArray_DATA(flux_scalings),
                            x0,
                            <double*> np.PyArray_DATA(dx0),
                            current_reservoir_id,
                            <int*> np.PyArray_DATA(variables_reservoir_id),
                            <int*> np.PyArray_DATA(variables_type_id),
                            <double*> np.PyArray_DATA(qstars),
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
               np.ndarray[double, ndim=2, mode='c'] input_mapping not None,
               np.ndarray[double, ndim=1, mode='c'] flux_scalings not None,
               double x0,
               np.ndarray[double, ndim=1, mode='c'] dx0 not None,
               np.ndarray[int, ndim=1, mode='c'] variables_reservoir_id not None,
               np.ndarray[int, ndim=1, mode='c'] variables_type_id not None,
               np.ndarray[double, ndim=2, mode='c'] qstars not None,
               np.ndarray[double, ndim=4, mode='c'] coefs_linear not None,
               np.ndarray[double, ndim=2, mode='c'] x1 not None,
               np.ndarray[double, ndim=3, mode='c'] dx1 not None,
               np.ndarray[double, ndim=2, mode='c'] fluxes not None,
               np.ndarray[double, ndim=4, mode='c'] dfluxes not None):

    # Check dimensions
    cdef int nreservoirs = alphas.shape[0]
    cdef int nalphas = alphas.shape[1]
    cdef int nval = qstars.shape[0]
    cdef int nvariables = dx0.shape[0]

    if gammas.shape[0] != nreservoirs:
        raise ValueError("gammas.shape[0] != nreservoirs")

    if flux_scalings.shape[0] != NFLUXES_INPUTS * nreservoirs:
        raise ValueError("flux_scalings.shape[0] != NFLUXES_INPUTS x nreservoirs")

    if variables_reservoir_id.shape[0] != nvariables:
        raise ValueError("variables_reservoir_id.shape[0] != nvariables")

    if variables_type_id.shape[0] != nvariables:
        raise ValueError("variables_type_id.shape[0] != nvariables")

    if qstars.shape[1] != NFLUXES_POTENTIAL * nreservoirs:
        raise ValueError("qstars.shape[1] != NFLUXES_POTENTIAL * nreservoirs")

    if input_mapping.shape[0] != NFLUXES_POTENTIAL * nreservoirs:
        raise ValueError("input_mapping.shape[0] != NFLUXES_POTENTIAL * nreservoirs")

    if input_mapping.shape[1] != NFLUXES_OUTPUTS * nreservoirs:
        raise ValueError("input_mapping.shape[1] != NFLUXES_OUTPUTS * nreservoirs")

    if coefs_linear.shape[0] != nreservoirs:
        raise ValueError("coefs_linear.shape[0] != nreservoirs")

    if coefs_linear.shape[1] != NFLUXES_INPUTS:
        raise ValueError("coefs_linear.shape[1] != NFLUXES_INPUTS")

    if coefs_linear.shape[2] != nalphas - 1:
        raise ValueError("coefs_linear.shape[2] != nalphas - 1")

    if coefs_linear.shape[3] != NFLUXES_POTENTIAL:
        raise ValueError("coefs_linear.shape[3] != NFLUXES_POTENTIAL")

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
                        <double*> np.PyArray_DATA(input_mapping),
                        <double*> np.PyArray_DATA(flux_scalings),
                        x0,
                        <double*> np.PyArray_DATA(dx0),
                        <int*> np.PyArray_DATA(variables_reservoir_id),
                        <int*> np.PyArray_DATA(variables_type_id),
                        <double*> np.PyArray_DATA(qstars),
                        <double*> np.PyArray_DATA(coefs_linear),
                        <double*> np.PyArray_DATA(x1),
                        <double*> np.PyArray_DATA(dx1),
                        <double*> np.PyArray_DATA(fluxes),
                        <double*> np.PyArray_DATA(dfluxes))
