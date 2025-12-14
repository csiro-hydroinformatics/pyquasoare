import warnings

import numpy as np

from pyquasoare import has_c_module

if has_c_module():
    import c_pyquasoare
else:
    raise ImportError("Cannot run rezeq without C module."
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
    ierr = c_pyquasoare.nonlinrouting(nsubdiv, timestep, theta, nu,
                                      q0, s0, inflows, outflows)
    if ierr > 0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.nonlinrouting returns {ierr} ({mess})")

    return outflows


def quadrouting(timestep, theta, q0, s0, inflows):
    inflows = np.array(inflows).astype(np.float64)
    outflows = np.zeros_like(inflows)
    ierr = c_pyquasoare.quadrouting(timestep, theta,
                                    q0, s0, inflows, outflows)
    if ierr > 0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.nonlinrouting returns {ierr} ({mess})")

    return outflows


# --- GR4J model ---
def gr4jprod_fluxes_noscaling(eta=1./2.25):
    """ GR4J production store fluxes: Ps, Es, Perc
    without climate input scaling.
    """
    def fpr(x):
        return np.where(x > 0, 1. - x**2, 0.)

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
    pi = max(0, (P-E)/X1)
    ei = max(0, (E-P)/X1)

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

    ierr = c_pyquasoare.gr4jprod(nsubdiv, X1, s0, inputs, outputs)

    if ierr > 0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.gr4jprod returns {ierr} ({mess})")

    return outputs
