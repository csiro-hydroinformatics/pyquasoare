import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_integ.h':
    double c_get_eps()

    double c_integrate_forward(double t0, double u0, double a, double b, double t)

    double c_integrate_inverse(double t0, double u0, double a, double b, double u)

    int c_find_alpha(int nalphas, double * alphas, double u0)

    int c_increment_fluxes(int nalphas, int nfluxes,
                            double * scalings,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            int jalpha,
                            double aoj,
                            double boj,
                            double t0,
                            double t1,
                            double u0,
                            double u1,
                            double * fluxes)

    int c_integrate(int nalphas, int nfluxes, double delta,
                                double * alphas,
                                double * scalings,
                                double * a_matrix_noscaling,
                                double * b_matrix_noscaling,
                                double u0,
                                double * final_u1,
                                double * fluxes)

    int c_run(int nalphas, int nfluxes, int nval, double delta,
                            double * alphas,
                            double * scalings,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double u0,
                            double * u1,
                            double * fluxes)


cdef extern from 'c_quadrouting.h':
    int c_quadrouting(int nval, double delta, double theta, double q0,
                        double s0, double *inflow, double * outflow)

def __cinit__(self):
    pass


def get_eps():
    return c_get_eps()

def integrate_forward(double t0, double u0, double a, double b, double t):
    return c_integrate_forward(t0, u0, a, b, t)


def integrate_inverse(double t0, double u0, double a, double b, double u):
    return c_integrate_inverse(t0, u0, a, b, u)


def find_alpha(double u0, np.ndarray[double, ndim=1, mode='c'] alphas not None):
    cdef int nalphas = alphas.shape[0]
    return c_find_alpha(nalphas, <double*> np.PyArray_DATA(alphas), u0)


def increment_fluxes(int jalpha, double aoj, double boj, \
                        double t0, double t1, \
                        double u0, double u1, \
                        np.ndarray[double, ndim=1, mode='c'] scalings not None,
                        np.ndarray[double, ndim=2, mode='c'] a_matrix_noscaling not None,
                        np.ndarray[double, ndim=2, mode='c'] b_matrix_noscaling not None,
                        np.ndarray[double, ndim=1, mode='c'] fluxes not None):
    # Check dimensions
    cdef int nalphas = a_matrix_noscaling.shape[0]+1
    cdef int nfluxes = a_matrix_noscaling.shape[1]

    if scalings.shape[0] != nfluxes:
        raise ValueError("scalings.shape[0] != nfluxes")

    if b_matrix_noscaling.shape[0] != nalphas-1:
        raise ValueError("b_matrix_noscaling.shape[0] != nalphas-1")

    if b_matrix_noscaling.shape[1] != nfluxes:
        raise ValueError("b_matrix_noscaling.shape[1] != nfluxes")

    # Run C code
    return c_increment_fluxes(nalphas, nfluxes,
                            <double*> np.PyArray_DATA(scalings),
                            <double*> np.PyArray_DATA(a_matrix_noscaling),
                            <double*> np.PyArray_DATA(b_matrix_noscaling),
                            jalpha, aoj, boj, t0, t1, u0, u1, \
                            <double*> np.PyArray_DATA(fluxes))


def integrate(double delta,  double u0, \
                        np.ndarray[double, ndim=1, mode='c'] alphas not None,\
                        np.ndarray[double, ndim=1, mode='c'] scalings not None,
                        np.ndarray[double, ndim=2, mode='c'] a_matrix_noscaling not None,
                        np.ndarray[double, ndim=2, mode='c'] b_matrix_noscaling not None,
                        np.ndarray[double, ndim=1, mode='c'] u1 not None,\
                        np.ndarray[double, ndim=1, mode='c'] fluxes not None):
    # Check dimensions
    cdef int nalphas = alphas.shape[0]
    cdef int nfluxes = a_matrix_noscaling.shape[1]

    if scalings.shape[0] != nfluxes:
        raise ValueError("scalings.shape[0] != nfluxes")

    if a_matrix_noscaling.shape[0] != nalphas-1:
        raise ValueError("a_matrix_noscaling.shape[0] != nalphas-1")

    if b_matrix_noscaling.shape[0] != nalphas-1:
        raise ValueError("b_matrix_noscaling.shape[0] != nalphas-1")

    if b_matrix_noscaling.shape[1] != nfluxes:
        raise ValueError("b_matrix_noscaling.shape[1] != nfluxes")

    if u1.shape[0] != 1:
        raise ValueError("u1.shape[0] != 1")

    if fluxes.shape[0] != nfluxes:
        raise ValueError("fluxes.shape[0] != nfluxes")

    # Run C code
    return c_integrate(nalphas, nfluxes, delta,
                                <double*> np.PyArray_DATA(alphas),
                                <double*> np.PyArray_DATA(scalings),
                                <double*> np.PyArray_DATA(a_matrix_noscaling),
                                <double*> np.PyArray_DATA(b_matrix_noscaling),
                                u0,
                                <double*> np.PyArray_DATA(u1),
                                <double*> np.PyArray_DATA(fluxes))


def run(double delta,  double u0, \
                        np.ndarray[double, ndim=1, mode='c'] alphas not None,\
                        np.ndarray[double, ndim=2, mode='c'] scalings not None,
                        np.ndarray[double, ndim=2, mode='c'] a_matrix_noscaling not None,
                        np.ndarray[double, ndim=2, mode='c'] b_matrix_noscaling not None,
                        np.ndarray[double, ndim=1, mode='c'] u1 not None,\
                        np.ndarray[double, ndim=2, mode='c'] fluxes not None):
    # Check dimensions
    cdef int nalphas = alphas.shape[0]
    cdef int nfluxes = a_matrix_noscaling.shape[1]
    cdef int nval = scalings.shape[0]

    if scalings.shape[1] != nfluxes:
        raise ValueError("scalings.shape[1] != nfluxes")

    if a_matrix_noscaling.shape[0] != nalphas-1:
        raise ValueError("a_matrix_noscaling.shape[0] != nalphas-1")

    if b_matrix_noscaling.shape[0] != nalphas-1:
        raise ValueError("b_matrix_noscaling.shape[0] != nalphas-1")

    if b_matrix_noscaling.shape[1] != nfluxes:
        raise ValueError("b_matrix_noscaling.shape[1] != nfluxes")

    if u1.shape[0] != nval:
        raise ValueError("u1.shape[0] != nval")

    if fluxes.shape[0] != nval:
        raise ValueError("fluxes.shape[0] != nval")

    if fluxes.shape[1] != nfluxes:
        raise ValueError("fluxes.shape[0] != nval")

    # Run C code
    return c_run(nalphas, nfluxes, nval, delta,
                                <double*> np.PyArray_DATA(alphas),
                                <double*> np.PyArray_DATA(scalings),
                                <double*> np.PyArray_DATA(a_matrix_noscaling),
                                <double*> np.PyArray_DATA(b_matrix_noscaling),
                                u0,
                                <double*> np.PyArray_DATA(u1),
                                <double*> np.PyArray_DATA(fluxes))


def quadrouting(double delta, double theta, double q0, double s0,
                    np.ndarray[double, ndim=1, mode='c'] inflows not None,
                    np.ndarray[double, ndim=1, mode='c'] outflows not None):

    cdef int nval = inflows.shape[0]
    if nval!=outflows.shape[0]:
        raise ValueError("inflows.shape[0]!=outflows.shape[0]")

    return c_quadrouting(nval, delta, theta, q0, s0,
                                <double*> np.PyArray_DATA(inflows),
                                <double*> np.PyArray_DATA(outflows))

