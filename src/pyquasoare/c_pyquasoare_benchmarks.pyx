import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_nonlinrouting.h':
    int c_quadrouting(int nval, double timestep,
                        double theta, double q0,
                        double s0, double *inflows, double * outflows)

    int c_nonlinrouting(int nval, int nsubdiv, double timestep,
                        double theta, double nu, double q0,
                        double s0, double *inflows, double * outflows)

cdef extern from 'c_gr4jprod.h':
    int c_gr4jprod(int nval, int nsubdiv, double X1, double s0,
                        double *inputs,
                        double * outputs)

cdef extern from 'c_gradbench.h':
    int c_gradbench_quadrouting(int nval, int nreaches,
                                double timestep,
                                double * x0_ro_route,
                                double * gammas_ro_route,
                                double * qin,
                                double * states_ro_route,
                                double * qout_ro_route)

    int c_gradbench_rainfallrunoff(int nval,
                                   double timestep,
                                   double x0_rr_prod,
                                   double x0_rr_route,
                                   double gamma_rr_prod,
                                   double gamma_rr_route,
                                   double demand_rr_route,
                                   double * pin,
                                   double * ein,
                                   double * states_rr_prod,
                                   double * states_rr_route,
                                   double * qout_rr_prod,
                                   double * qout_rr_route)

    int c_gradbench_semidist(int nval, int nreaches,
                             double timestep,
                             double x0_rr_prod,
                             double x0_rr_route,
                             double x0_ro_route,
                             double * areas,
                             double * gammas_rr_prod,
                             double * gammas_rr_route,
                             double * gammas_ro_route,
                             double * demands_rr_route,
                             double * pin,
                             double * ein,
                             double * states_rr_prod,
                             double * states_rr_route,
                             double * states_ro_route,
                             double * qout_rr_prod,
                             double * qout_rr_route,
                             double * qout_ro_route)


def __cinit__(self):
    pass


def quadrouting(double timestep, double theta, \
                    double q0, double s0,
                    np.ndarray[double, ndim=1, mode='c'] inflows not None,
                    np.ndarray[double, ndim=1, mode='c'] outflows not None):

    cdef int nval = inflows.shape[0]
    if nval!=outflows.shape[0]:
        raise ValueError("inflows.shape[0]!=outflows.shape[0]")

    return c_quadrouting(nval, timestep, theta, q0, s0,
                                <double*> np.PyArray_DATA(inflows),
                                <double*> np.PyArray_DATA(outflows))


def nonlinrouting(int nsubdiv, double timestep, double theta, double nu, \
                    double q0, double s0,
                    np.ndarray[double, ndim=1, mode='c'] inflows not None,
                    np.ndarray[double, ndim=1, mode='c'] outflows not None):

    cdef int nval = inflows.shape[0]
    if nval!=outflows.shape[0]:
        raise ValueError("inflows.shape[0]!=outflows.shape[0]")

    return c_nonlinrouting(nval, nsubdiv, timestep, theta, nu, q0, s0,
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
    if outputs.shape[1]!=6:
        raise ValueError("outputs.shape[1]!=6")

    return c_gr4jprod(nval, nsubdiv, X1, s0,
                                <double*> np.PyArray_DATA(inputs),
                                <double*> np.PyArray_DATA(outputs))


def gradbench_quadrouting(double timestep,
             np.ndarray[double, ndim=1, mode='c'] x0_ro_route not None,
             np.ndarray[double, ndim=1, mode='c'] gammas_ro_route not None,
             np.ndarray[double, ndim=1, mode='c'] qin not None,
             np.ndarray[double, ndim=2, mode='c'] states_ro_route not None,
             np.ndarray[double, ndim=2, mode='c'] qout_ro_route not None):

    cdef int nval = qin.shape[0]
    cdef int nreaches = x0_ro_route.shape[0]

    if states_ro_route.shape[0] != nval:
        raise ValueError("states_ro_route.shape[0] != qin.shape[0]")

    if qout_ro_route.shape[0] != nval:
        raise ValueError("qout_ro_route.shape[0] != qin.shape[0]")

    if gammas_ro_route.shape[0] != nreaches:
        raise ValueError("gammas_ro_route.shape[0] != x0_ro_route.shape[0]")

    if states_ro_route.shape[1] != nreaches:
        raise ValueError("states_ro_route.shape[1] != x0_ro_route.shape[0]")

    if qout_ro_route.shape[1] != nreaches:
        raise ValueError("qout_ro_route.shape[1] != x0_ro_route.shape[0]")

    return c_gradbench_quadrouting(nval, nreaches, timestep,
                          <double*> np.PyArray_DATA(x0_ro_route),
                          <double*> np.PyArray_DATA(gammas_ro_route),
                          <double*> np.PyArray_DATA(qin),
                          <double*> np.PyArray_DATA(states_ro_route),
                          <double*> np.PyArray_DATA(qout_ro_route))


def gradbench_rainfallrunoff(double timestep,
                             double x0_rr_prod,
                             double x0_rr_route,
                             double gamma_rr_prod,
                             double gamma_rr_route,
                             double demand_rr_route,
                             np.ndarray[double, ndim=1, mode='c'] pin not None,
                             np.ndarray[double, ndim=1, mode='c'] ein not None,
                             np.ndarray[double, ndim=1, mode='c'] states_rr_prod not None,
                             np.ndarray[double, ndim=1, mode='c'] states_rr_route not None,
                             np.ndarray[double, ndim=1, mode='c'] qout_rr_prod not None,
                             np.ndarray[double, ndim=1, mode='c'] qout_rr_route not None):

    cdef int nval = pin.shape[0]

    if ein.shape[0] != nval:
        raise ValueError("ein.shape[0] != pin.shape[0]")

    if states_rr_prod.shape[0] != nval:
        raise ValueError("states_rr_prod.shape[0] != pin.shape[0]")

    if states_rr_route.shape[0] != nval:
        raise ValueError("states_rr_route.shape[0] != pin.shape[0]")

    if qout_rr_prod.shape[0] != nval:
        raise ValueError("qout_rr_prod.shape[0] != pin.shape[0]")

    if qout_rr_route.shape[0] != nval:
        raise ValueError("qout_rr_route.shape[0] != pin.shape[0]")

    return c_gradbench_rainfallrunoff(nval, timestep,
                                      x0_rr_prod,
                                      x0_rr_route,
                                      gamma_rr_prod,
                                      gamma_rr_route,
                                      demand_rr_route,
                                      <double*> np.PyArray_DATA(pin),
                                      <double*> np.PyArray_DATA(ein),
                                      <double*> np.PyArray_DATA(states_rr_prod),
                                      <double*> np.PyArray_DATA(states_rr_route),
                                      <double*> np.PyArray_DATA(qout_rr_prod),
                                      <double*> np.PyArray_DATA(qout_rr_route))


def gradbench_semidist(double timestep,
                       double x0_rr_prod,
                       double x0_rr_route,
                       double x0_ro_route,
                       np.ndarray[double, ndim=1, mode='c'] areas not None,
                       np.ndarray[double, ndim=1, mode='c'] gammas_rr_prod not None,
                       np.ndarray[double, ndim=1, mode='c'] gammas_rr_route not None,
                       np.ndarray[double, ndim=1, mode='c'] gammas_ro_route not None,
                       np.ndarray[double, ndim=1, mode='c'] demands_rr_route not None,
                       np.ndarray[double, ndim=2, mode='c'] pin not None,
                       np.ndarray[double, ndim=2, mode='c'] ein not None,
                       np.ndarray[double, ndim=2, mode='c'] states_rr_prod not None,
                       np.ndarray[double, ndim=2, mode='c'] states_rr_route not None,
                       np.ndarray[double, ndim=2, mode='c'] states_ro_route not None,
                       np.ndarray[double, ndim=2, mode='c'] qout_rr_prod not None,
                       np.ndarray[double, ndim=2, mode='c'] qout_rr_route not None,
                       np.ndarray[double, ndim=2, mode='c'] qout_ro_route not None):

    cdef int nval = pin.shape[0]
    cdef int nreaches = areas.shape[0]

    if gammas_rr_route.shape[0] != nreaches:
        raise ValueError("gammas_rr_route.shape[0] != areas.shape[0]")

    if demands_rr_route.shape[0] != nreaches:
        raise ValueError("demands_rr_route.shape[0] != areas.shape[0]")

    if gammas_ro_route.shape[0] != nreaches:
        raise ValueError("gammas_ro_route.shape[0] != areas.shape[0]")

    if pin.shape[1] != nreaches:
        raise ValueError("pin.shape[1] != areas.shape[0]")

    if ein.shape[0] != nval:
        raise ValueError("ein.shape[0] != pin.shape[0]")

    if ein.shape[1] != nreaches:
        raise ValueError("ein.shape[1] != areas.shape[0]")

    if states_rr_prod.shape[0] != nval:
        raise ValueError("states_rr_prod.shape[0] != pin.shape[0]")

    if states_rr_prod.shape[1] != nreaches:
        raise ValueError("states_rr_prod.shape[1] != areas.shape[0]")

    if states_rr_route.shape[0] != nval:
        raise ValueError("states_rr_route.shape[0] != pin.shape[0]")

    if states_rr_route.shape[1] != nreaches:
        raise ValueError("states_rr_route.shape[1] != areas.shape[0]")

    if states_ro_route.shape[0] != nval:
        raise ValueError("states_ro_route.shape[0] != pin.shape[0]")

    if states_ro_route.shape[1] != nreaches:
        raise ValueError("states_rr_route.shape[1] != areas.shape[0]")

    if qout_rr_prod.shape[0] != nval:
        raise ValueError("qout_rr_prod.shape[0] != pin.shape[0]")

    if qout_rr_prod.shape[1] != nreaches:
        raise ValueError("qout_rr_prod.shape[1] != areas.shape[0]")

    if qout_rr_route.shape[0] != nval:
        raise ValueError("qout_rr_route.shape[0] != pin.shape[0]")

    if qout_rr_route.shape[1] != nreaches:
        raise ValueError("qout_rr_route.shape[1] != areas.shape[0]")

    if qout_ro_route.shape[0] != nval:
        raise ValueError("qout_ro_route.shape[0] != pin.shape[0]")

    if qout_ro_route.shape[1] != nreaches:
        raise ValueError("qout_ro_route.shape[1] != areas.shape[0]")

    return c_gradbench_semidist(nval, nreaches, timestep,
                                x0_rr_prod,
                                x0_rr_route,
                                x0_ro_route,
                                <double*> np.PyArray_DATA(areas),
                                <double*> np.PyArray_DATA(gammas_rr_prod),
                                <double*> np.PyArray_DATA(gammas_rr_route),
                                <double*> np.PyArray_DATA(gammas_ro_route),
                                <double*> np.PyArray_DATA(demands_rr_route),
                                <double*> np.PyArray_DATA(pin),
                                <double*> np.PyArray_DATA(ein),
                                <double*> np.PyArray_DATA(states_rr_prod),
                                <double*> np.PyArray_DATA(states_rr_route),
                                <double*> np.PyArray_DATA(states_ro_route),
                                <double*> np.PyArray_DATA(qout_rr_prod),
                                <double*> np.PyArray_DATA(qout_rr_route),
                                <double*> np.PyArray_DATA(qout_ro_route))


