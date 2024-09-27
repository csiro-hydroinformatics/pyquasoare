import numpy as np

from pyquasoare import has_c_module
if has_c_module():
    import c_pyquasoare
else:
    raise ImportError("Cannot run rezeq without C module. Please compile C code")

# --- Non linear routing model ---
def nonlinrouting_fluxes_noscaling(nu):
    """ fluxes from reservoir dS/dt = inflow-q0*(S/theta)**nu
        or d(S/theta)/dt = inflow/theta-q0/theta*(S/theta)**nu

        Consequently non scaled fluxes are
        f1 = 1. (inflow to be multiplied by inflow/theta)
        f2 = -x^nu (outflow to be multiplied by q0/theta)
    """
    assert nu>1
    fin = lambda x: 1.
    fout = lambda x: -x**nu if x>=0 else 0.
    fluxes = [fin, fout]

    dfin = lambda x: 0.
    dfout = lambda x: -nu*x**(nu-1.) if x>=0 else 0.
    dfluxes = [dfin, dfout]

    return fluxes, dfluxes


def nonlinrouting(nsubdiv, timestep, theta, nu, q0, s0, inflows):
    inflows = np.array(inflows).astype(np.float64)
    outflows = np.zeros_like(inflows)
    ierr = c_pyquasoare.nonlinrouting(nsubdiv, timestep, theta, nu, \
                                    q0, s0, inflows, outflows)
    if ierr>0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.nonlinrouting returns {ierr} ({mess})")

    return outflows


def quadrouting(timestep, theta, q0, s0, inflows):
    inflows = np.array(inflows).astype(np.float64)
    outflows = np.zeros_like(inflows)
    ierr = c_pyquasoare.quadrouting(timestep, theta, \
                                    q0, s0, inflows, outflows)
    if ierr>0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.nonlinrouting returns {ierr} ({mess})")

    return outflows


# --- GR4J model ---
def gr4jprod_fluxes_noscaling(eta=1./2.25):
    """ GR4J production store fluxes: Ps, Es, Perc without climate input scaling """
    fpr = lambda x: (1.-x**2) if x>0 else 1.
    fae = lambda x: -x*(2.-x) if x<1. else -1.
    fperc = lambda x: -(eta*x)**5/4./eta if x>0 else 0.
    fluxes = [fpr, fae, fperc]

    dfpr = lambda x: -2.*x
    dfae = lambda x: -2.*(1.-x)
    dfperc = lambda x: -5./4.*(eta*x)**4
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

    fpr = lambda x: pi*normf[0](x)
    fae = lambda x: ei*normf[1](x)
    fperc = lambda x: normf[2](x)
    fluxes = [fpr, fae, fperc]

    dfpr = lambda x: pi*dnormf[0](x)
    dfae = lambda x: ei*dnormf[1](x)
    dfperc = lambda x: dnormf[2](x)
    dfluxes = [dfpr, dfae, dfperc]

    return fluxes, dfluxes



def gr4jprod(nsubdiv, X1, s0, inputs):
    inputs = np.ascontiguousarray(inputs).astype(np.float64)
    # Outputs variables = S, PR, AE, PERC, PR, AE
    outputs = np.zeros((len(inputs), 6))

    ierr = c_pyquasoare.gr4jprod(nsubdiv, X1, s0, inputs, outputs)

    if ierr>0:
        mess = c_pyquasoare.get_error_message(ierr).decode()
        raise ValueError(f"c_pyquasoare.gr4jprod returns {ierr} ({mess})")

    return outputs




