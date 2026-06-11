import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_quasoare_grad.h':

    int c_grad_integrate(int nalphas, int nvariables,
                         double * alphas,
                         double timestep,
                         double Qstars[2],
                         double gamma,
                         double flux_scalings[3],
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
                         double fluxes[6]);

    int c_grad_reservoir(int nalphas, int nval, int nvariables,
                        int errors,
                        double timestep, double gamma,
                        double flux_scalings[3],
                        double xmin, double xmax,
                        double x0,
                        double * dx0,
                        int current_reservoir_id,
                        int *variables_reservoir_id,
                        int *variables_type_id,
                        double * alphas,
                        double * Qstars,
                        double * coefs_linear,
                        int * niter,
                        double * x1,
                        double * dx1,
                        double * fluxes);

    int c_grad_model(int nreservoirs, int nalphas, int nval,
                     int nvariables, int errors,
                     double timestep,
                     double x0,
                     double * dx0,
                     int *variables_reservoir_id,
                     int *variables_type_id,
                     double * Qstars,
                     double * Qstars_mapping,
                     double * alphas,
                     double * gammas,
                     double * flux_scalings,
                     double * coefs_linear,
                     double * x1,
                     double * dx1,
                     double * fluxes);

def __cinit__(self):
    pass


def grad_integrate(int current_reservoir_id,
                   np.ndarray[double, ndim=1, mode='c'] alphas not None,
                   double timestep,
                   np.ndarray[double, ndim=1, mode='c'] qstars not None,
                   double gamma,
                   np.ndarray[double, ndim=1, mode='c'] flux_scalings not None,
                   np.ndarray[double, ndim=3, mode='c'] coefs_linear not None,
                   double t0, double x0,
                   np.ndarray[double, ndim=1, mode='c'] dx0 not None,
                   np.ndarray[int, ndim=1, mode='c'] variables_reservoir_id not None,
                   np.ndarray[int, ndim=1, mode='c'] variables_type_id not None,
                   np.ndarray[int, ndim=1, mode='c'] niter not None,
                   np.ndarray[double, ndim=1, mode='c'] x1 not None,
                   np.ndarray[double, ndim=1, mode='c'] dx1 not None,
                   np.ndarray[double, ndim=1, mode='c'] fluxes not None):

    # Check dimensions
    cdef int nalphas = alphas.shape[0]
    cdef int nvariables = dx0.shape[0]

    if qstars.shape[0] != 2:
        raise ValueError("qstars.shape[0] != 2")

    if flux_scalings.shape[0] != 3:
        raise ValueError("coefs_linear.shape[0] != 3")

    if coefs_linear.shape[0] != 3:
        raise ValueError("coefs_linear.shape[0] != 3")

    if coefs_linear.shape[1] != nalphas - 1:
        raise ValueError("coefs_linear.shape[1] != nalphas - 1")

    if coefs_linear.shape[2] != 2:
        raise ValueError("coefs_linear.shape[2] != 2")

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

    if fluxes.shape[0] != 6:
        raise ValueError("fluxes.shape[0] != 6")

    # Run C code
    return c_grad_integrate(nalphas, nvariables,
                           <double*> np.PyArray_DATA(alphas),
                           timestep,
                           <double*> np.PyArray_DATA(qstars),
                           gamma,
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
                           <double*> np.PyArray_DATA(fluxes))


def grad_reservoir(int errors, double timestep, double gamma,
                   np.ndarray[double, ndim=1, mode='c'] flux_scalings not None,
                   double xmin, double xmax, double x0,
                   np.ndarray[double, ndim=1, mode='c'] dx0 not None,
                   int current_reservoir_id,
                   np.ndarray[int, ndim=1, mode='c'] variables_reservoir_id not None,
                   np.ndarray[int, ndim=1, mode='c'] variables_type_id not None,
                   np.ndarray[double, ndim=1, mode='c'] alphas not None,
                   np.ndarray[double, ndim=2, mode='c'] qstars not None,
                   np.ndarray[double, ndim=3, mode='c'] coefs_linear not None,
                   np.ndarray[int, ndim=1, mode='c'] niter not None,
                   np.ndarray[double, ndim=1, mode='c'] x1 not None,
                   np.ndarray[double, ndim=1, mode='c'] dx1 not None,
                   np.ndarray[double, ndim=2, mode='c'] fluxes not None):

    # Check dimensions
    cdef int nalphas = alphas.shape[0]
    cdef int nval = qstars.shape[0]
    cdef int nvariables = dx0.shape[0]

    if flux_scalings.shape[0] != 3:
        raise ValueError("flux_scalings.shape[0] != 3")

    if variables_reservoir_id.shape[0] != nvariables:
        raise ValueError("variables_reservoir_id.shape[0] != nvariables")

    if variables_type_id.shape[0] != nvariables:
        raise ValueError("variables_type_id.shape[0] != nvariables")

    if qstars.shape[1] != 2:
        raise ValueError("qstars.shape[1] != 2")

    if coefs_linear.shape[0] != 3:
        raise ValueError("coefs_linear.shape[0] != 3")

    if coefs_linear.shape[1] != nalphas - 1:
        raise ValueError("coefs_linear.shape[1] != nalphas - 1")

    if coefs_linear.shape[2] != 2:
        raise ValueError("coefs_linear.shape[2] != 2")

    if x1.shape[0] != nval:
        raise ValueError("x1.shape[0] != nval")

    if dx1.shape[0] != nval:
        raise ValueError("dx1.shape[0] != nval")

    if niter.shape[0] != nval:
        raise ValueError("niter.shape[0] != nval")

    if fluxes.shape[0] != nval:
        raise ValueError("fluxes.shape[0] != nval")

    if fluxes.shape[1] != 6:
        raise ValueError("fluxes.shape[0] != 6")

    # Run C code
    return c_grad_reservoir(nalphas, nval, nvariables, errors,
                            timestep, gamma,
                            <double*> np.PyArray_DATA(flux_scalings),
                            xmin, xmax, x0,
                            <double*> np.PyArray_DATA(dx0),
                            current_reservoir_id,
                            <int*> np.PyArray_DATA(variables_reservoir_id),
                            <int*> np.PyArray_DATA(variables_type_id),
                            <double*> np.PyArray_DATA(alphas),
                            <double*> np.PyArray_DATA(qstars),
                            <double*> np.PyArray_DATA(coefs_linear),
                            <int*> np.PyArray_DATA(niter),
                            <double*> np.PyArray_DATA(x1),
                            <double*> np.PyArray_DATA(dx1),
                            <double*> np.PyArray_DATA(fluxes))


def grad_model(int errors, double timestep, double x0,
               np.ndarray[double, ndim=1, mode='c'] dx0 not None,
               np.ndarray[int, ndim=1, mode='c'] variables_reservoir_id not None,
               np.ndarray[int, ndim=1, mode='c'] variables_type_id not None,
               np.ndarray[double, ndim=2, mode='c'] alphas not None,
               np.ndarray[double, ndim=2, mode='c'] qstars not None,
               np.ndarray[double, ndim=2, mode='c'] qstars_mapping not None,
               np.ndarray[double, ndim=1, mode='c'] gammas not None,
               np.ndarray[double, ndim=1, mode='c'] flux_scalings not None,
               np.ndarray[double, ndim=4, mode='c'] coefs_linear not None,
               np.ndarray[double, ndim=2, mode='c'] x1 not None,
               np.ndarray[double, ndim=3, mode='c'] dx1 not None,
               np.ndarray[double, ndim=2, mode='c'] fluxes not None):

    # Check dimensions
    cdef int nreservoirs = alphas.shape[0]
    cdef int nalphas = alphas.shape[1]
    cdef int nval = qstars.shape[0]
    cdef int nvariables = dx0.shape[0]

    if gammas.shape[0] != nreservoirs:
        raise ValueError("gammas.shape[0] != nreservoirs")

    if flux_scalings.shape[0] != 3 * nreservoirs:
        raise ValueError("flux_scalings.shape[0] != 3 x nreservoirs")

    if variables_reservoir_id.shape[0] != nvariables:
        raise ValueError("variables_reservoir_id.shape[0] != nvariables")

    if variables_type_id.shape[0] != nvariables:
        raise ValueError("variables_type_id.shape[0] != nvariables")

    if qstars.shape[1] != 2 * nreservoirs:
        raise ValueError("qstars.shape[1] != 2 * nreservoirs")

    if qstars_mapping.shape[0] != 2 * nreservoirs:
        raise ValueError("qstars_mapping.shape[0] != 2 * nreservoirs")

    if qstars_mapping.shape[1] != 6 * nreservoirs:
        raise ValueError("qstars_mapping.shape[1] != 6 * nreservoirs")

    if coefs_linear.shape[0] != nreservoirs:
        raise ValueError("coefs_linear.shape[0] != nreservoirs")

    if coefs_linear.shape[1] != 3:
        raise ValueError("coefs_linear.shape[1] != 3")

    if coefs_linear.shape[2] != nalphas - 1:
        raise ValueError("coefs_linear.shape[2] != nalphas - 1")

    if coefs_linear.shape[3] != 2:
        raise ValueError("coefs_linear.shape[3] != 2")

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

    if fluxes.shape[1] != 6 * nreservoirs:
        raise ValueError("fluxes.shape[0] != 6 * nreservoirs")

    # Run C code
    return c_grad_model(nreservoirs, nalphas, nval, nvariables, errors,
                       timestep, x0,
                       <double*> np.PyArray_DATA(dx0),
                       <int*> np.PyArray_DATA(variables_reservoir_id),
                       <int*> np.PyArray_DATA(variables_type_id),
                       <double*> np.PyArray_DATA(qstars),
                       <double*> np.PyArray_DATA(qstars_mapping),
                       <double*> np.PyArray_DATA(alphas),
                       <double*> np.PyArray_DATA(gammas),
                       <double*> np.PyArray_DATA(flux_scalings),
                       <double*> np.PyArray_DATA(coefs_linear),
                       <double*> np.PyArray_DATA(x1),
                       <double*> np.PyArray_DATA(dx1),
                       <double*> np.PyArray_DATA(fluxes))
