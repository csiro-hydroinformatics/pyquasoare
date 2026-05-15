import warnings

import numpy as np

from pyquasoare import has_c_module

if has_c_module():
    import c_pyquasoare
else:
    raise ImportError("Cannot run without C module."
                      + " Please compile C code")

def quadgrad(timestep, qin, x0, gammas):
    timestep = np.float64(timestep)
    x0 = np.float64(x0)
    gammas = np.ascontiguousarray(gammas).astype(np.float64)
    qin = np.ascontiguousarray(qin).astype(np.float64)

    nval = qin.shape[0]
    nreaches = len(gammas)
    states = np.empty((nval, nreaches))
    qout = np.empty((nval, nreaches))

    ierr = c_pyquasoare.quadgrad(timestep, x0, gammas, qin, states, qout)

    if ierr > 0:
        raise ValueError(f"c_pyquasoare.quadgrad returns {ierr}")

    return states, qout

def quadgrad_tangent(timestep, qin, x0, gammas, gammas_diff):
    timestep = np.float64(timestep)
    x0 = np.float64(x0)
    gammas = np.ascontiguousarray(gammas).astype(np.float64)
    gammas_diff = np.ascontiguousarray(gammas_diff).astype(np.float64)
    qin = np.ascontiguousarray(qin).astype(np.float64)

    nval = qin.shape[0]
    nreaches = len(gammas)
    states = np.empty((nval, nreaches))
    states_diff = np.empty((nval, nreaches))
    qout = np.empty((nval, nreaches))
    qout_diff = np.empty((nval, nreaches))

    ierr = c_pyquasoare.quadgrad_tangent(timestep, x0, gammas,
                                      gammas_diff,
                                      qin,
                                      states, states_diff,
                                      qout, qout_diff)

    if ierr > 0:
        raise ValueError(f"c_pyquasoare.quadgrad_tangent returns {ierr}")

    return states, qout, states_diff, qout_diff
