import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_rezeq_utils.h':
    double c_get_eps()
    double c_get_atol()
    double c_get_rtol()
    double c_get_inf()
    double c_get_nan()
    int c_get_nfluxes_max()
    double c_compiler_accuracy_kahan()
    int c_discrimin(double a, double b, double c, double discr[2])

    int c_find_alpha(int nalphas, double * alphas, double s0)

    int c_get_error_message(int err_code, char message[100])


cdef extern from 'c_rezeq_quad.h':
    double c_quad_fun(double a, double b, double c, double s)
    double c_quad_grad(double a, double b, double c, double s)

    int c_quad_steady(double a, double b, double c, double steady[2])

    int c_quad_coefficients(int islin, double a0, double a1,
                                double f0, double f1, double fm,
                                double coefs[3])

    double c_quad_delta_t_max(double a, double b, double c,
                                double Delta, double qD, double s0);

    double c_quad_forward(double a, double b, double c,
                                double Delta, double qD,
                                double t0, double s0, double t);

    double c_quad_inverse(double a, double b, double c,
                                double Delta, double qD,
                                double s0, double s1);

    int c_quad_fluxes(int nfluxes,
                            double * aj_vector,
                            double * bj_vector,
                            double * cj_vector,
                            double aoj, double boj, double coj,
                            double t0, double t1, double s0, double s1,
                            double * fluxes)

    int c_quad_integrate(int nalphas, int nfluxes,
                                double * alphas, double * scalings,
                                double * a_matrix_noscaling,
                                double * b_matrix_noscaling,
                                double * c_matrix_noscaling,
                                double t0,
                                double s0,
                                double delta,
                                int *niter, double * s1, double * fluxes)

    int c_quad_model(int nalphas, int nfluxes, int nval, double delta,
                            double * alphas, double * scalings,
                            double * a_matrix_noscaling,
                            double * b_matrix_noscaling,
                            double * c_matrix_noscaling,
                            double s0, int * niter,
                            double * s1, double * fluxes);


cdef extern from 'c_nonlinrouting.h':
    int c_nonlinrouting(int nval, int nsubdiv, double delta,
                        double theta, double nu, double q0,
                        double s0, double *inflows, double * outflows)

cdef extern from 'c_gr4jprod.h':
    int c_gr4jprod(int nval, int nsubdiv, double X1, double s0,
                        double *inputs,
                        double * outputs)

def __cinit__(self):
    pass

def get_eps():
    return c_get_eps()

def get_atol():
    return c_get_atol()

def get_rtol():
    return c_get_rtol()

def get_nan():
    return c_get_nan()

def get_inf():
    return c_get_inf()

def get_nfluxes_max():
    return c_get_nfluxes_max()

def compiler_accuracy_kahan():
    return c_compiler_accuracy_kahan()


def discrimin(double a, double b, double c, \
                np.ndarray[double, ndim=1, mode='c'] discr not None):
    assert discr.shape[0] == 2
    return c_discrimin(a, b, c, <double*> np.PyArray_DATA(discr))


def get_error_message(int err_code):
    cdef char message[100]
    c_get_error_message(err_code, message)
    return message


def quad_fun(double a, double b, double c, double s):
    return c_quad_fun(a, b, c, s)

def quad_fun_vect(np.ndarray[double, ndim=1, mode='c'] a not None,\
                np.ndarray[double, ndim=1, mode='c'] b not None,\
                np.ndarray[double, ndim=1, mode='c'] c not None,\
                np.ndarray[double, ndim=1, mode='c'] s not None,\
                np.ndarray[double, ndim=1, mode='c'] o not None):
    cdef int nval = a.shape[0]
    assert nval == b.shape[0]
    assert nval == c.shape[0]
    assert nval == s.shape[0]
    assert nval == o.shape[0]

    cdef int k
    for k in range(nval):
        o[k] = c_quad_fun(a[k], b[k], c[k], s[k])

    return 0


def quad_grad(double a, double b, double c, double s):
    return c_quad_grad(a, b, c, s)

def quad_grad_vect(np.ndarray[double, ndim=1, mode='c'] a not None,\
                np.ndarray[double, ndim=1, mode='c'] b not None,\
                np.ndarray[double, ndim=1, mode='c'] c not None,\
                np.ndarray[double, ndim=1, mode='c'] s not None,\
                np.ndarray[double, ndim=1, mode='c'] o not None):
    cdef int nval = a.shape[0]
    assert nval == b.shape[0]
    assert nval == c.shape[0]
    assert nval == s.shape[0]
    assert nval == o.shape[0]

    cdef int k
    for k in range(nval):
        o[k] = c_quad_grad(a[k], b[k], c[k], s[k])

    return 0



def quad_coefficients(int islin, double a0, double a1,
                        double f0, double f1, double fm,
                        np.ndarray[double, ndim=1, mode='c'] coefs):
    assert coefs.shape[0] == 3
    return c_quad_coefficients(islin, a0, a1, f0, f1, fm,
                                <double*> np.PyArray_DATA(coefs))


def quad_steady(double a, double b, double c, \
                        np.ndarray[double, ndim=1, mode='c'] steady not None):
    assert steady.shape[0] == 2
    return c_quad_steady(a, b, c, <double*> np.PyArray_DATA(steady))


def quad_steady_vect(np.ndarray[double, ndim=1, mode='c'] a not None, \
                        np.ndarray[double, ndim=1, mode='c'] b not None, \
                        np.ndarray[double, ndim=1, mode='c'] c not None, \
                        np.ndarray[double, ndim=2, mode='c'] steady not None):
    cdef int k
    cdef int nval = a.shape[0]
    assert b.shape[0] == nval
    assert c.shape[0] == nval
    assert steady.shape[0] == nval
    assert steady.shape[1] == 2

    for k in range(nval):
        c_quad_steady(a[k], b[k], c[k], <double*> np.PyArray_DATA(steady[k]))

    return 0


def quad_forward(double a, double b, double c, \
                        double Delta, double qD, \
                        double t0, double s0, double t):
    return c_quad_forward(a, b, c, Delta, qD, t0, s0, t)


def quad_forward_vect(double a, double b, double c, \
                        double Delta, double qD, \
                        double t0, double s0, \
                        np.ndarray[double, ndim=1, mode='c'] t not None,\
                        np.ndarray[double, ndim=1, mode='c'] s not None):
    cdef int nval = t.shape[0]
    cdef int i
    assert s.shape[0] == nval
    for i in range(nval):
        s[i] = c_quad_forward(a, b, c, Delta, qD, t0, s0, t[i])
    return 0


def quad_delta_t_max(double a, double b, double c, \
                        double Delta, double qD, double s0):
    return c_quad_delta_t_max(a, b, c, Delta, qD, s0)


def quad_inverse(double a, double b, double c, \
                        double Delta, double qD, \
                        double s0, double s1):
    return c_quad_inverse(a, b, c, Delta, qD, s0, s1)


def quad_inverse_vect(double a, double b, double c, double Delta, double qD, double s0, \
                            np.ndarray[double, ndim=1, mode='c'] s1 not None,\
                            np.ndarray[double, ndim=1, mode='c'] t not None):
    cdef int nval = s1.shape[0]
    cdef int i
    assert t.shape[0] == nval
    for i in range(nval):
            t[i] = c_quad_inverse(a, b, c, Delta, qD, s0, s1[i])
    return 0


def find_alpha(np.ndarray[double, ndim=1, mode='c'] alphas not None,\
                double s0):
    cdef int nalphas = alphas.shape[0]
    return c_find_alpha(nalphas, <double*> np.PyArray_DATA(alphas), s0)


def quad_fluxes(np.ndarray[double, ndim=1, mode='c'] aj_vector not None,
                    np.ndarray[double, ndim=1, mode='c'] bj_vector not None,
                    np.ndarray[double, ndim=1, mode='c'] cj_vector not None,
                    double aoj, double boj, double coj, \
                    double t0, double t1, \
                    double s0, double s1, \
                    np.ndarray[double, ndim=1, mode='c'] fluxes not None):
    # Check dimensions
    cdef int nfluxes = aj_vector.shape[0]

    if bj_vector.shape[0] != nfluxes:
        raise ValueError("bj_vector.shape[0] != nfluxes")

    if cj_vector.shape[0] != nfluxes:
        raise ValueError("cj_vector.shape[0] != nfluxes")

    # Run C code
    return c_quad_fluxes(nfluxes,
                            <double*> np.PyArray_DATA(aj_vector),
                            <double*> np.PyArray_DATA(bj_vector),
                            <double*> np.PyArray_DATA(cj_vector),
                            aoj, boj, coj, t0, t1, s0, s1, \
                            <double*> np.PyArray_DATA(fluxes))


def quad_integrate(np.ndarray[double, ndim=1, mode='c'] alphas not None,\
                        np.ndarray[double, ndim=1, mode='c'] scalings not None,
                        np.ndarray[double, ndim=2, mode='c'] a_matrix_noscaling not None,
                        np.ndarray[double, ndim=2, mode='c'] b_matrix_noscaling not None,
                        np.ndarray[double, ndim=2, mode='c'] c_matrix_noscaling not None,
                        double t0, double s0, double delta, \
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
    return c_quad_integrate(nalphas, nfluxes,
                                <double*> np.PyArray_DATA(alphas),
                                <double*> np.PyArray_DATA(scalings),
                                <double*> np.PyArray_DATA(a_matrix_noscaling),
                                <double*> np.PyArray_DATA(b_matrix_noscaling),
                                <double*> np.PyArray_DATA(c_matrix_noscaling),
                                t0, s0, delta,
                                <int*> np.PyArray_DATA(niter),
                                <double*> np.PyArray_DATA(s1),
                                <double*> np.PyArray_DATA(fluxes))


def quad_model(double delta, \
        np.ndarray[double, ndim=1, mode='c'] alphas not None,\
        np.ndarray[double, ndim=2, mode='c'] scalings not None,
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
    return c_quad_model(nalphas, nfluxes, nval, delta,
                                <double*> np.PyArray_DATA(alphas),
                                <double*> np.PyArray_DATA(scalings),
                                <double*> np.PyArray_DATA(a_matrix_noscaling),
                                <double*> np.PyArray_DATA(b_matrix_noscaling),
                                <double*> np.PyArray_DATA(c_matrix_noscaling),
                                s0,
                                <int*> np.PyArray_DATA(niter),
                                <double*> np.PyArray_DATA(s1),
                                <double*> np.PyArray_DATA(fluxes))


def nonlinrouting(int nsubdiv, double delta, double theta, double nu, \
                    double q0, double s0,
                    np.ndarray[double, ndim=1, mode='c'] inflows not None,
                    np.ndarray[double, ndim=1, mode='c'] outflows not None):

    cdef int nval = inflows.shape[0]
    if nval!=outflows.shape[0]:
        raise ValueError("inflows.shape[0]!=outflows.shape[0]")

    return c_nonlinrouting(nval, nsubdiv, delta, theta, nu, q0, s0,
                                <double*> np.PyArray_DATA(inflows),
                                <double*> np.PyArray_DATA(outflows))


def gr4jprod(int nsubdiv, double X1, double s0,
                    np.ndarray[double, ndim=2, mode='c'] inputs not None,
                    np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int nval = inputs.shape[0]
    if inputs.shape[1]!=2:
        raise ValueError("inputs.shape[1]!=2")
    if nval!=outputs.shape[0]:
        raise ValueError("rain.shape[0]!=outputs.shape[0]")
    if outputs.shape[1]!=4:
        raise ValueError("outputs.shape[1]!=4")

    return c_gr4jprod(nval, nsubdiv, X1, s0,
                                <double*> np.PyArray_DATA(inputs),
                                <double*> np.PyArray_DATA(outputs))

