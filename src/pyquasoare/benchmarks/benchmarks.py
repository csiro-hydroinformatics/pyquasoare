import warnings

import numpy as np

from pyquasoare import has_c_module

if has_c_module():
    import c_pyquasoare_benchmarks
else:
    raise ImportError("Cannot run benchmarks without C module."
                      + " Please compile C code")


# --- Non linear routing model ---
def nonlinrouting_fluxes_noscaling(nu):
    """ fluxes from reservoir dS/dt = inflow-q0*(S/theta)**nu
        or d(S/theta)/dt = inflow/theta-q0/theta*(S/theta)**nu

        Consequently non scaled fluxes are
        f1 = 1. (inflow to be multiplied by inflow/theta)
        f2 = -x^nu (outflow to be multiplied by q0/theta)
    """
    if nu < 1:
        warnmess = "nu is lower than 1, flux functions "\
            + "are not Lipschitz continuous"
        warnings.warn(warnmess)

    def fin(x):
        return np.ones_like(x)

    def fout(x):
        return np.where(x >= 0, -x**nu, 0.)

    fluxes = [fin, fout]

    def dfin(x):
        return 0.

    def dfout(x):
        return np.where(x >= 0, -nu * x**(nu - 1.), 0.)

    dfluxes = [dfin, dfout]

    return fluxes, dfluxes


def nonlinrouting(nsubdiv, timestep, theta, nu, q0, s0, inflows):
    inflows = np.array(inflows).astype(np.float64)
    outflows = np.zeros_like(inflows)
    ierr = c_pyquasoare_benchmarks.nonlinrouting(nsubdiv, timestep, theta, nu,
                                      q0, s0, inflows, outflows)
    if ierr > 0:
        raise ValueError(f"c_pyquasoare_benchmarks.nonlinrouting returns {ierr}.")

    return outflows


def quadrouting(timestep, theta, q0, s0, inflows):
    inflows = np.array(inflows).astype(np.float64)
    outflows = np.zeros_like(inflows)
    ierr = c_pyquasoare_benchmarks.quadrouting(timestep, theta,
                                    q0, s0, inflows, outflows)
    if ierr > 0:
        raise ValueError(f"c_pyquasoare_benchmarks.nonlinrouting returns {ierr}.")

    return outflows


# --- GR4J model ---
def gr4jprod_fluxes_noscaling(eta=1./2.25):
    """ GR4J production store fluxes: Ps, Es, Perc
    without climate input scaling.
    """
    def fpr(x):
        return np.where(x > 0, 1. - x**2, 1.)

    def fae(x):
        return np.where(x < 1, -x * (2 - x), -1.)

    def fperc(x):
        return np.where(x > 0, -(eta * x)**5 / 4. / eta, 0.)

    fluxes = [fpr, fae, fperc]

    def dfpr(x):
        return np.where(x > 0, -2 * x, 0.)

    def dfae(x):
        return np.where(x < 1, -2 * (1. - x), 0.)

    def dfperc(x):
        return np.where(x > 0, -5./4.*(eta*x)**4, 0.)

    dfluxes = [dfpr, dfae, dfperc]

    return fluxes, dfluxes


def gr4jprod_fluxes_scaled(P, E, X1, eta=1./2.25):
    """ GR4J production store fluxes: Ps, Es, Perc with climate input scaling
        Pi = max(P-E, 0)/X1
        Ei = max(E-P, 0)/X1
    """
    # interception reservoir
    pi = max(0, (P-E) / X1)
    ei = max(0, (E-P) / X1)

    # normalised fluxes
    normf, dnormf = gr4jprod_fluxes_noscaling(eta)

    def fpr(x):
        return pi*normf[0](x)

    def fae(x):
        return ei*normf[1](x)

    def fperc(x):
        return normf[2](x)

    fluxes = [fpr, fae, fperc]

    def dfpr(x):
        return pi*dnormf[0](x)

    def dfae(x):
        return ei*dnormf[1](x)

    def dfperc(x):
        return dnormf[2](x)

    dfluxes = [dfpr, dfae, dfperc]

    return fluxes, dfluxes


def gr4jprod(nsubdiv, X1, s0, inputs):
    inputs = np.ascontiguousarray(inputs).astype(np.float64)
    # Outputs variables = S, PR, AE, PERC, PR, AE
    outputs = np.zeros((len(inputs), 6))

    ierr = c_pyquasoare_benchmarks.gr4jprod(nsubdiv, X1, s0, inputs, outputs)

    if ierr > 0:
        raise ValueError(f"c_pyquasoare_benchmarks.gr4jprod returns {ierr}")

    return outputs


def gradbench_linrouting(timestep, qin, x0, thetas):
    timestep = np.float64(timestep)
    thetas = np.ascontiguousarray(thetas).astype(np.float64)
    qin = np.ascontiguousarray(qin).astype(np.float64)

    nval = qin.shape[0]
    nreaches = len(thetas)
    x0 = np.array([x0] * nreaches)
    states = np.empty((nval, nreaches))
    qout = np.empty((nval, nreaches))

    ierr = c_pyquasoare_benchmarks.gradbench_linrouting(timestep, x0,
                                                        thetas, qin,
                                                        states, qout)
    if ierr > 0:
        raise ValueError(f"c_pyquasoare_benchmarks.gradbench_linrouting returns {ierr}")

    return states, qout


def gradbench_linrouting_tangent(timestep, qin, x0, thetas, dthetas):
    timestep = np.float64(timestep)
    thetas = np.ascontiguousarray(thetas).astype(np.float64)
    dthetas = np.ascontiguousarray(dthetas).astype(np.float64)
    qin = np.ascontiguousarray(qin).astype(np.float64)

    nval = qin.shape[0]
    nreaches = len(thetas)
    x0 = np.array([x0] * nreaches)
    states = np.empty((nval, nreaches))
    dstates = np.empty((nval, nreaches))
    qout = np.empty((nval, nreaches))
    dqout = np.empty((nval, nreaches))

    ierr = c_pyquasoare_benchmarks.gradbench_linrouting_d(timestep, x0,
                                                          thetas,
                                                          dthetas,
                                                          qin,
                                                          states,
                                                          dstates,
                                                          qout,
                                                          dqout)
    if ierr > 0:
        raise ValueError(f"c_pyquasoare_benchmarks.gradbench_linrouting_d returns {ierr}")

    return states, dstates, qout, dqout


def gradbench_quadrouting(timestep, qin, x0, gammas, out=None):
    timestep = np.float64(timestep)
    gammas = np.ascontiguousarray(gammas).astype(np.float64)
    qin = np.ascontiguousarray(qin).astype(np.float64)

    nval = qin.shape[0]
    nreaches = len(gammas)
    x0 = np.array([x0] * nreaches)

    if out is None:
        states = np.empty((nval, nreaches))
        qout = np.empty((nval, nreaches))
    else:
        states, qout = out

    ierr = c_pyquasoare_benchmarks.gradbench_quadrouting(timestep, x0,
                                                         gammas, qin,
                                                         states, qout)
    if ierr > 0:
        raise ValueError(f"c_pyquasoare_benchmarks.gradbench_quadrouting returns {ierr}")

    return states, qout


def gradbench_quadrouting_tangent(timestep, qin, x0, gammas, dgammas, out=None):
    timestep = np.float64(timestep)
    gammas = np.ascontiguousarray(gammas).astype(np.float64)
    dgammas = np.ascontiguousarray(dgammas).astype(np.float64)
    qin = np.ascontiguousarray(qin).astype(np.float64)

    nval = qin.shape[0]
    nreaches = len(gammas)
    x0 = np.array([x0] * nreaches)

    if out is None:
        states = np.empty((nval, nreaches))
        dstates = np.empty((nval, nreaches))
        qout = np.empty((nval, nreaches))
        dqout = np.empty((nval, nreaches))
    else:
        states, dstates, qout, dqout = out

    ierr = c_pyquasoare_benchmarks.gradbench_quadrouting_d(timestep, x0,
                                                           gammas,
                                                           dgammas,
                                                           qin,
                                                           states,
                                                           dstates,
                                                           qout,
                                                           dqout)
    if ierr > 0:
        raise ValueError(f"c_pyquasoare_benchmarks.gradbench_quadrouting_d returns {ierr}")

    return states, dstates, qout, dqout


def gradbench_rainfall_runoff(timestep, pin, ein, x0,
                             gamma_rr_prod, gamma_rr_route,
                             demand_rr_route):
    timestep = np.float64(timestep)
    gamma_rr_prod = np.float64(gamma_rr_prod)
    gamma_rr_route = np.float64(gamma_rr_route)
    demand_rr_route = np.float64(demand_rr_route)
    pin = np.ascontiguousarray(pin).astype(np.float64)
    ein = np.ascontiguousarray(ein).astype(np.float64)

    nval = pin.shape[0]
    x0_rr_prod = x0
    x0_rr_route = x0
    states_rr_prod = np.empty(nval)
    states_rr_route = np.empty(nval)
    qout_rr_prod = np.empty(nval)
    qout_rr_route = np.empty(nval)

    ierr = c_pyquasoare_benchmarks.gradbench_rainfall_runoff(timestep,
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
        raise ValueError(f"c_pyquasoare_benchmarks.gradbench_quadrouting returns {ierr}")

    return states_rr_prod, states_rr_route, \
        qout_rr_prod, qout_rr_route


def gradbench_rainfall_runoff_tangent(timestep, pin, ein, x0,
                                      gamma_rr_prod,
                                      dgamma_rr_prod,
                                      gamma_rr_route,
                                      dgamma_rr_route,
                                      demand_rr_route,
                                      ddemand_rr_route):
    timestep = np.float64(timestep)
    gamma_rr_prod = np.float64(gamma_rr_prod)
    dgamma_rr_prod = np.float64(dgamma_rr_prod)
    gamma_rr_route = np.float64(gamma_rr_route)
    dgamma_rr_route = np.float64(dgamma_rr_route)
    demand_rr_route = np.float64(demand_rr_route)
    ddemand_rr_route = np.float64(ddemand_rr_route)
    pin = np.ascontiguousarray(pin).astype(np.float64)
    ein = np.ascontiguousarray(ein).astype(np.float64)

    nval = pin.shape[0]
    x0_rr_prod = x0
    x0_rr_route = x0
    states_rr_prod = np.empty(nval)
    dstates_rr_prod = np.empty(nval)
    states_rr_route = np.empty(nval)
    dstates_rr_route = np.empty(nval)
    qout_rr_prod = np.empty(nval)
    dqout_rr_prod = np.empty(nval)
    qout_rr_route = np.empty(nval)
    dqout_rr_route = np.empty(nval)

    ierr = c_pyquasoare_benchmarks.gradbench_rainfall_runoff_d(timestep,
                                                               x0_rr_prod,
                                                               x0_rr_route,
                                                               gamma_rr_prod,
                                                               dgamma_rr_prod,
                                                               gamma_rr_route,
                                                               dgamma_rr_route,
                                                               demand_rr_route,
                                                               ddemand_rr_route,
                                                               pin, ein,
                                                               states_rr_prod,
                                                               dstates_rr_prod,
                                                               states_rr_route,
                                                               dstates_rr_route,
                                                               qout_rr_prod,
                                                               dqout_rr_prod,
                                                               qout_rr_route,
                                                               dqout_rr_route)
    if ierr > 0:
        raise ValueError(f"c_pyquasoare_benchmarks.gradbench_rainfall_runoff_d returns {ierr}")

    return states_rr_prod, dstates_rr_prod, \
        states_rr_route, dstates_rr_route, \
        qout_rr_prod, dqout_rr_prod, \
        qout_rr_route, dqout_rr_route


def gradbench_semidist(timestep, pin, ein, x0, areas,
                       gammas_rr_prod, gammas_rr_route,
                       gammas_ro_route, demands_rr_route,
                       out=None):
    timestep = np.float64(timestep)
    areas = areas.astype(np.float64)
    gammas_rr_prod = gammas_rr_prod.astype(np.float64)
    gammas_rr_route = gammas_rr_route.astype(np.float64)
    gammas_ro_route = gammas_ro_route.astype(np.float64)
    demands_rr_route = demands_rr_route.astype(np.float64)
    pin = np.ascontiguousarray(pin).astype(np.float64)
    ein = np.ascontiguousarray(ein).astype(np.float64)

    nreaches = areas.shape[0]

    nval = pin.shape[0]
    x0 = np.float64(x0)
    x0_rr_prod = x0
    x0_rr_route = x0
    x0_ro_route = x0

    if out is None:
        states_rr_prod = np.empty((nval, nreaches))
        states_rr_route = np.empty((nval, nreaches))
        states_ro_route = np.empty((nval, nreaches))
        qout_rr_prod =  np.empty((nval, nreaches))
        qout_rr_route = np.empty((nval, nreaches))
        qout_ro_route = np.empty((nval, nreaches))
    else:
        states_rr_prod, states_rr_route, \
        states_ro_route, qout_rr_prod, \
        qout_rr_route, qout_ro_route = out

    ierr = c_pyquasoare_benchmarks.gradbench_semidist(timestep,
                                                      x0_rr_prod,
                                                      x0_rr_route,
                                                      x0_ro_route,
                                                      areas,
                                                      gammas_rr_prod,
                                                      gammas_rr_route,
                                                      gammas_ro_route,
                                                      demands_rr_route,
                                                      pin, ein,
                                                      states_rr_prod,
                                                      states_rr_route,
                                                      states_ro_route,
                                                      qout_rr_prod,
                                                      qout_rr_route,
                                                      qout_ro_route)
    if ierr > 0:
        raise ValueError(f"c_pyquasoare_benchmarks.gradbench_semidist returns {ierr}")

    return states_rr_prod, states_rr_route, states_ro_route, \
        qout_rr_prod, qout_rr_route, qout_ro_route


def gradbench_semidist_tangent(timestep, pin, ein, x0, areas,
                               gammas_rr_prod,
                               dgammas_rr_prod,
                               gammas_rr_route,
                               dgammas_rr_route,
                               gammas_ro_route,
                               dgammas_ro_route,
                               demands_rr_route,
                               ddemands_rr_route,
                               out=None):
    timestep = np.float64(timestep)
    areas = areas.astype(np.float64)
    gammas_rr_prod = gammas_rr_prod.astype(np.float64)
    dgammas_rr_prod = dgammas_rr_prod.astype(np.float64)
    gammas_rr_route = gammas_rr_route.astype(np.float64)
    dgammas_rr_route = dgammas_rr_route.astype(np.float64)
    gammas_ro_route = gammas_ro_route.astype(np.float64)
    dgammas_ro_route = dgammas_ro_route.astype(np.float64)
    demands_rr_route = demands_rr_route.astype(np.float64)
    ddemands_rr_route = ddemands_rr_route.astype(np.float64)
    pin = np.ascontiguousarray(pin).astype(np.float64)
    ein = np.ascontiguousarray(ein).astype(np.float64)

    nreaches = areas.shape[0]

    nval = pin.shape[0]
    x0 = np.float64(x0)
    x0_rr_prod = x0
    x0_rr_route = x0
    x0_ro_route = x0
    if out is None:
        states_rr_prod = np.empty((nval, nreaches))
        dstates_rr_prod = np.empty((nval, nreaches))
        states_rr_route = np.empty((nval, nreaches))
        dstates_rr_route = np.empty((nval, nreaches))
        states_ro_route = np.empty((nval, nreaches))
        dstates_ro_route = np.empty((nval, nreaches))
        qout_rr_prod =  np.empty((nval, nreaches))
        dqout_rr_prod =  np.empty((nval, nreaches))
        qout_rr_route = np.empty((nval, nreaches))
        dqout_rr_route = np.empty((nval, nreaches))
        qout_ro_route = np.empty((nval, nreaches))
        dqout_ro_route = np.empty((nval, nreaches))
    else:
        states_rr_prod, dstates_rr_prod, \
        states_rr_route, dstates_rr_route, \
        states_ro_route, dstates_ro_route, \
        qout_rr_prod, dqout_rr_prod, \
        qout_rr_route, dqout_rr_route, \
        qout_ro_route, dqout_ro_route = out

    ierr = c_pyquasoare_benchmarks.gradbench_semidist_d(timestep,
                                                        x0_rr_prod,
                                                        x0_rr_route,
                                                        x0_ro_route,
                                                        areas,
                                                        gammas_rr_prod,
                                                        dgammas_rr_prod,
                                                        gammas_rr_route,
                                                        dgammas_rr_route,
                                                        gammas_ro_route,
                                                        dgammas_ro_route,
                                                        demands_rr_route,
                                                        ddemands_rr_route,
                                                        pin, ein,
                                                        states_rr_prod,
                                                        dstates_rr_prod,
                                                        states_rr_route,
                                                        dstates_rr_route,
                                                        states_ro_route,
                                                        dstates_ro_route,
                                                        qout_rr_prod,
                                                        dqout_rr_prod,
                                                        qout_rr_route,
                                                        dqout_rr_route,
                                                        qout_ro_route,
                                                        dqout_ro_route)
    if ierr > 0:
        raise ValueError(f"c_pyquasoare_benchmarks.gradbench_semidist_d returns {ierr}")

    return states_rr_prod, \
        dstates_rr_prod, \
        states_rr_route, \
        dstates_rr_route, \
        states_ro_route, \
        dstates_ro_route, \
        qout_rr_prod, \
        dqout_rr_prod, \
        qout_rr_route, \
        dqout_rr_route, \
        qout_ro_route, \
        dqout_ro_route
