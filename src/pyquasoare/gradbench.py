import numpy as np

from pyquasoare import has_c_module

if has_c_module():
    import c_pyquasoare
else:
    raise ImportError("Cannot run without C module."
                      + " Please compile C code")


def gradbench_quadrouting(timestep, qin, x0, gammas):
    timestep = np.float64(timestep)
    gammas = np.ascontiguousarray(gammas).astype(np.float64)
    qin = np.ascontiguousarray(qin).astype(np.float64)

    nval = qin.shape[0]
    nreaches = len(gammas)
    x0 = np.array([x0] * nreaches)
    states = np.empty((nval, nreaches))
    qout = np.empty((nval, nreaches))

    ierr = c_pyquasoare.gradbench_quadrouting(timestep, x0,
                                              gammas, qin,
                                              states, qout)
    if ierr > 0:
        raise ValueError(f"c_pyquasoare.gradbench_quadrouting returns {ierr}")

    return states, qout


def gradbench_rainfallrunoff(timestep, pin, ein, x0,
                             gamma_rr_prod, gamma_rr_route,
                             demand_rr_route):
    timestep = np.float64(timestep)
    pin = np.ascontiguousarray(pin).astype(np.float64)
    ein = np.ascontiguousarray(ein).astype(np.float64)

    nval = pin.shape[0]
    x0_rr_prod = x0
    x0_rr_route = x0
    states_rr_prod = np.empty(nval)
    states_rr_route = np.empty(nval)
    qout_rr_prod = np.empty(nval)
    qout_rr_route = np.empty(nval)

    ierr = c_pyquasoare.gradbench_rainfallrunoff(timestep,
                                                 x0_rr_prod,
                                                 x0_rr_route,
                                                 gamma_rr_prod,
                                                 gamma_rr_route,
                                                 demand_rr_route,
                                                 pin, ein,
                                                 states_rr_prod,
                                                 states_rr_route,
                                                 qout_rr_prod,
                                                 qout_rr_route)
    if ierr > 0:
        raise ValueError(f"c_pyquasoare.gradbench_quadrouting returns {ierr}")

    return states_rr_prod, states_rr_route, \
        qout_rr_prod, qout_rr_route


#def gradbench_tangent(timestep, qin, x0, gammas, gammas_diff):
#    timestep = np.float64(timestep)
#    x0 = np.float64(x0)
#    gammas = np.ascontiguousarray(gammas).astype(np.float64)
#    gammas_diff = np.ascontiguousarray(gammas_diff).astype(np.float64)
#    qin = np.ascontiguousarray(qin).astype(np.float64)
#
#    nval = qin.shape[0]
#    nreaches = len(gammas)
#    states = np.empty((nval, nreaches))
#    states_diff = np.empty((nval, nreaches))
#    qout = np.empty((nval, nreaches))
#    qout_diff = np.empty((nval, nreaches))
#
#    ierr = c_pyquasoare.gradbench_tangent(timestep, x0, gammas,
#                                      gammas_diff,
#                                      qin,
#                                      states, states_diff,
#                                      qout, qout_diff)
#
#    if ierr > 0:
#        raise ValueError(f"c_pyquasoare.gradbench_tangent returns {ierr}")
#
#    return states, qout, states_diff, qout_diff
