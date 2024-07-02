import math
import numpy as np

from scipy.optimize import minimize
from scipy.integrate import solve_ivp

from pyrezeq.rezeq import REZEQ_EPS

def integrate_forward_numerical(funs, dfuns, t0, s0, t, \
                            method="Radau", max_step=np.inf, \
                            fun_args=None):
    nfluxes = len(funs)
    assert len(dfuns) == nfluxes
    v = np.zeros(nfluxes)
    def fun_ivp(t, y):
        for i in range(nfluxes):
            v[i] = funs[i](y[0])
        return v

    m = np.zeros((nfluxes, nfluxes))
    jac_ivp = None
    if method == "Radau":
        def jac_ivp(t, y):
            for i in range(nfluxes):
                m[i, 0] = dfuns[i](y[0])
            return m

    res = solve_ivp(\
            fun=fun_ivp, \
            t_span=[t0, t[-1]], \
            y0=s0, \
            method=method, \
            max_step=max_step, \
            jac=jac_ivp, \
            t_eval=t, \
            args=fun_args)

    return res.t, res.y.T.squeeze()


def routing_numerical(delta, theta, q0, s0, \
                        inflows, exponent=2, \
                        method="Radau", \
                        max_step_frac=np.inf):
    def fun(t, y, qi):
        return qi-q0*(y/theta)**exponent

    def dfun(t, y, qi):
        return np.array([-q0*nu*(y/theta)**(exponent-1)])

    nval = len(inflows)
    fluxes = []
    store = []
    for t in range(nval):
        qi = inflows.iloc[t]
        res = integrate_forward_numerical(fun, dfun, 0, s0, delta, \
                            method=method, max_step=delta*max_step_frac, \
                            fun_args=(qi, ))
        raise ValueError("TODO")
        s1 = res.y[0][-1]
        fluxes.append((s0-s1)/delta+qi)
        store.append(s1)

        # Loop
        s0 = s1

    return np.array(store), np.array(fluxes)
