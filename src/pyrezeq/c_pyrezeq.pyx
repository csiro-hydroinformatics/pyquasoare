import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_utils.h':
    double c_get_eps()

    int c_find_alpha(int nalphas, double * alphas, double s0);

    int c_get_error_message(int err_code, char message[100]);


cdef extern from 'c_integ.h':
    double c_approx_fun(double nu, double a, double b, double c, double s);

    double c_approx_jac(double nu, double a, double b, double c, double s);

    double c_integrate_forward(double nu, double a, double b, double c,
                                    double t0, double s0, double t);
    double c_integrate_delta_t_max(double nu, double a, double b, double c, double s0);

    double c_integrate_inverse(double nu, double a, double b, double c,
                                        double s0, double s1);

    int c_increment_fluxes(int nfluxes,
                            double * scalings,
                            double nu,
                            double * aj_vector_noscaling,
                            double * bj_vector_noscaling,
                            double * cj_vector_noscaling,
                            double aoj,
                            double boj,
                            double coj,
                            double t0,
                            double t1,
                            double s0,
                            double s1,
                            double * fluxes);

    int c_integrate(int nalphas, int nfluxes, double delta,
                                double * alphas,
                                double * scalings,
                                double * nu_vector,
                                double * a_matrix_noscaling,
                                double * b_matrix_noscaling,
                                double * c_matrix_noscaling,
                                double s0,
                                int * niter,
                                double * s1,
                                double * fluxes);

cdef extern from 'c_run.h':
    int c_run(int nalphas, int nfluxes, int nval, double delta,
                                double * alphas,
                                double * scalings,
                                double * nu_vector,
                                double * a_matrix_noscaling,
                                double * b_matrix_noscaling,
                                double * c_matrix_noscaling,
                                double s0,
                                int * niter,
                                double * s1,
                                double * fluxes);

cdef extern from 'c_steady.h':
    int c_steady_state(double nu, double a, double b, double c, double steady[2]);


cdef extern from 'c_quadrouting.h':
    int c_quadrouting(int nval, double delta, double theta, double q0,
                        double s0, double *inflow, double * outflow)

def __cinit__(self):
    pass


def get_eps():
    return c_get_eps()


def get_error_message(int err_code):
    cdef char message[100]
    c_get_error_message(err_code, message)
    return message


def approx_fun(double nu, double a, double b, double c, double s):
    return c_approx_fun(nu, a, b, c, s)


def approx_jac(double nu, double a, double b, double c, double s):
    return c_approx_jac(nu, a, b, c, s)


def approx_fun_vect(double nu, double a, double b, double c, \
                np.ndarray[double, ndim=1, mode='c'] s not None,\
                np.ndarray[double, ndim=1, mode='c'] ds not None):
    cdef int nval = s.shape[0]
    cdef int k

    for k in range(nval):
        ds[k] = c_approx_fun(nu, a, b, c, s[k])

    return 0

def approx_jac_vect(double nu, double a, double b, double c, \
                np.ndarray[double, ndim=1, mode='c'] s not None,\
                np.ndarray[double, ndim=1, mode='c'] ds not None):
    cdef int nval = s.shape[0]
    cdef int k
    assert ds.shape[0] == nval
    for k in range(nval):
        ds[k] = c_approx_jac(nu, a, b, c, s[k])

    return 0


def integrate_forward(double nu, double a, double b, double c, \
                        double t0, double s0, double t):
    return c_integrate_forward(nu, a, b, c, t0, s0, t)


def integrate_forward_vect(double nu, double a, double b, double c, \
                        double t0, double s0, \
                        np.ndarray[double, ndim=1, mode='c'] t not None,\
                        np.ndarray[double, ndim=1, mode='c'] s not None):
    cdef int nval = t.shape[0]
    cdef int i
    assert s.shape[0] == nval
    for i in range(nval):
        s[i] = c_integrate_forward(nu, a, b, c, t0, s0, t[i])
    return 0


def steady_state(double nu, double a, double b, double c, \
                        np.ndarray[double, ndim=1, mode='c'] steady not None):
    assert steady.shape[0] == 2
    return c_steady_state(nu, a, b, c, \
                            <double*> np.PyArray_DATA(steady))


def steady_state_vect(np.ndarray[double, ndim=1, mode='c'] nus not None,\
                        np.ndarray[double, ndim=1, mode='c'] a not None, \
                        np.ndarray[double, ndim=1, mode='c'] b not None, \
                        np.ndarray[double, ndim=1, mode='c'] c not None, \
                        np.ndarray[double, ndim=2, mode='c'] steady not None):
    cdef int k
    cdef int nval = nus.shape[0]
    assert a.shape[0] == nval
    assert b.shape[0] == nval
    assert c.shape[0] == nval
    assert steady.shape[0] == nval
    assert steady.shape[1] == 2

    for k in range(nval):
        c_steady_state(nus[k], a[k], b[k], c[k], \
                            <double*> np.PyArray_DATA(steady[k]))
    return 0


def integrate_delta_t_max(double nu, double a, double b, double c, double s0):
    return c_integrate_delta_t_max(nu, a, b, c, s0)


def integrate_inverse(double nu, double a, double b, double c, \
                            double s0, double s1):
    return c_integrate_inverse(nu, a, b, c, s0, s1)


def integrate_inverse_vect(double nu, double a, double b, double c, \
                            double s0, \
                            np.ndarray[double, ndim=1, mode='c'] s1 not None,\
                            np.ndarray[double, ndim=1, mode='c'] t not None):
    cdef int nval = s1.shape[0]
    cdef int i
    assert t.shape[0] == nval
    for i in range(nval):
            t[i] = c_integrate_inverse(nu, a, b, c, s0, s1[i])
    return 0


def find_alpha(np.ndarray[double, ndim=1, mode='c'] alphas not None,\
                double s0):
    cdef int nalphas = alphas.shape[0]
    return c_find_alpha(nalphas, <double*> np.PyArray_DATA(alphas), s0)


def increment_fluxes(np.ndarray[double, ndim=1, mode='c'] scalings not None,
                    double nu, \
                    np.ndarray[double, ndim=1, mode='c'] aj_vector_noscaling not None,
                    np.ndarray[double, ndim=1, mode='c'] bj_vector_noscaling not None,
                    np.ndarray[double, ndim=1, mode='c'] cj_vector_noscaling not None,
                    double aoj, double boj, double coj, \
                    double t0, double t1, \
                    double s0, double s1, \
                    np.ndarray[double, ndim=1, mode='c'] fluxes not None):
    # Check dimensions
    cdef int nfluxes = aj_vector_noscaling.shape[0]

    if scalings.shape[0] != nfluxes:
        raise ValueError("scalings.shape[0] != nfluxes")

    if bj_vector_noscaling.shape[0] != nfluxes:
        raise ValueError("bj_vector_noscaling.shape[0] != nfluxes")

    if cj_vector_noscaling.shape[0] != nfluxes:
        raise ValueError("cj_vector_noscaling.shape[0] != nfluxes")

    # Run C code
    return c_increment_fluxes(nfluxes,
                            <double*> np.PyArray_DATA(scalings),
                            nu,
                            <double*> np.PyArray_DATA(aj_vector_noscaling),
                            <double*> np.PyArray_DATA(bj_vector_noscaling),
                            <double*> np.PyArray_DATA(cj_vector_noscaling),
                            aoj, boj, coj, t0, t1, s0, s1, \
                            <double*> np.PyArray_DATA(fluxes))


def integrate(double delta,
                        np.ndarray[double, ndim=1, mode='c'] alphas not None,\
                        np.ndarray[double, ndim=1, mode='c'] scalings not None,
                        np.ndarray[double, ndim=1, mode='c'] nu_vector not None,
                        np.ndarray[double, ndim=2, mode='c'] a_matrix_noscaling not None,
                        np.ndarray[double, ndim=2, mode='c'] b_matrix_noscaling not None,
                        np.ndarray[double, ndim=2, mode='c'] c_matrix_noscaling not None,
                        double s0, \
                        np.ndarray[int, ndim=1, mode='c'] niter not None,\
                        np.ndarray[double, ndim=1, mode='c'] s1 not None,\
                        np.ndarray[double, ndim=1, mode='c'] fluxes not None):
    # Check dimensions
    cdef int nalphas = alphas.shape[0]
    cdef int nfluxes = a_matrix_noscaling.shape[1]

    if scalings.shape[0] != nfluxes:
        raise ValueError("scalings.shape[0] != nfluxes")

    if b_matrix_noscaling.shape[1] != nfluxes:
        raise ValueError("b_matrix_noscaling.shape[1] != nfluxes")

    if c_matrix_noscaling.shape[1] != nfluxes:
        raise ValueError("c_matrix_noscaling.shape[1] != nfluxes")

    if nu_vector.shape[0] != nalphas-1:
        raise ValueError("nu_vector.shape[0] != nalphas-1")

    if a_matrix_noscaling.shape[0] != nalphas-1:
        raise ValueError("a_matrix_noscaling.shape[0] != nalphas-1")

    if b_matrix_noscaling.shape[0] != nalphas-1:
        raise ValueError("b_matrix_noscaling.shape[0] != nalphas-1")

    if c_matrix_noscaling.shape[0] != nalphas-1:
        raise ValueError("c_matrix_noscaling.shape[0] != nalphas-1")

    if s1.shape[0] != 1:
        raise ValueError("s1.shape[0] != 1")

    if fluxes.shape[0] != nfluxes:
        raise ValueError("fluxes.shape[0] != nfluxes")

    # Run C code
    return c_integrate(nalphas, nfluxes, delta,
                                <double*> np.PyArray_DATA(alphas),
                                <double*> np.PyArray_DATA(scalings),
                                <double*> np.PyArray_DATA(nu_vector),
                                <double*> np.PyArray_DATA(a_matrix_noscaling),
                                <double*> np.PyArray_DATA(b_matrix_noscaling),
                                <double*> np.PyArray_DATA(c_matrix_noscaling),
                                s0,
                                <int*> np.PyArray_DATA(niter),
                                <double*> np.PyArray_DATA(s1),
                                <double*> np.PyArray_DATA(fluxes))


def run(double delta, \
        np.ndarray[double, ndim=1, mode='c'] alphas not None,\
        np.ndarray[double, ndim=2, mode='c'] scalings not None,
        np.ndarray[double, ndim=1, mode='c'] nu_vector not None,
        np.ndarray[double, ndim=2, mode='c'] a_matrix_noscaling not None,
        np.ndarray[double, ndim=2, mode='c'] b_matrix_noscaling not None,
        np.ndarray[double, ndim=2, mode='c'] c_matrix_noscaling not None,
        double s0, \
        np.ndarray[int, ndim=1, mode='c'] niter not None,\
        np.ndarray[double, ndim=1, mode='c'] s1 not None,\
        np.ndarray[double, ndim=2, mode='c'] fluxes not None):

    # Check dimensions
    cdef int nalphas = alphas.shape[0]
    cdef int nfluxes = a_matrix_noscaling.shape[1]
    cdef int nval = scalings.shape[0]

    if scalings.shape[0] != nfluxes:
        raise ValueError("scalings.shape[0] != nfluxes")

    if b_matrix_noscaling.shape[1] != nfluxes:
        raise ValueError("b_matrix_noscaling.shape[1] != nfluxes")

    if c_matrix_noscaling.shape[1] != nfluxes:
        raise ValueError("c_matrix_noscaling.shape[1] != nfluxes")

    if nu_vector.shape[0] != nalphas-1:
        raise ValueError("nu_vector.shape[0] != nalphas-1")

    if a_matrix_noscaling.shape[0] != nalphas-1:
        raise ValueError("a_matrix_noscaling.shape[0] != nalphas-1")

    if b_matrix_noscaling.shape[0] != nalphas-1:
        raise ValueError("b_matrix_noscaling.shape[0] != nalphas-1")

    if c_matrix_noscaling.shape[0] != nalphas-1:
        raise ValueError("c_matrix_noscaling.shape[0] != nalphas-1")

    if s1.shape[0] != nval:
        raise ValueError("s1.shape[0] != nval")

    if niter.shape[0] != nval:
        raise ValueError("niter.shape[0] != nval")

    if fluxes.shape[0] != nval:
        raise ValueError("fluxes.shape[0] != nval")

    if fluxes.shape[1] != nfluxes:
        raise ValueError("fluxes.shape[0] != nval")

    # Run C code
    return c_run(nalphas, nfluxes, nval, delta,
                                <double*> np.PyArray_DATA(alphas),
                                <double*> np.PyArray_DATA(scalings),
                                <double*> np.PyArray_DATA(nu_vector),
                                <double*> np.PyArray_DATA(a_matrix_noscaling),
                                <double*> np.PyArray_DATA(b_matrix_noscaling),
                                <double*> np.PyArray_DATA(c_matrix_noscaling),
                                s0,
                                <int*> np.PyArray_DATA(niter),
                                <double*> np.PyArray_DATA(s1),
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

