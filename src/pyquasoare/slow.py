import warnings
import math
import numpy as np

from scipy.integrate import solve_ivp

from pyquasoare.approx import QUASOARE_EPS, QUASOARE_PI, \
                                QUASOARE_ATOL, QUASOARE_RTOL
from pyquasoare.approx import isequal, notequal, isnull, notnull
from pyquasoare.integrate import quad_constants
from pyquasoare.models import ERRORS


def integrate_numerical(fluxes, dfluxes, t0, s0, t,
                        method="Radau", max_step=np.inf,
                        scaling=None, v=None, m=None):
    nfluxes = len(fluxes)
    assert len(dfluxes) == nfluxes
    multi_fluxes = nfluxes > 1

    # ODE derivative - can be initialised outside to save time
    v = np.zeros(nfluxes+multi_fluxes) if v is None else v

    def fun_ivp(t, y):
        total = 0.
        for i in range(nfluxes):
            s = 1. if scaling is None else scaling[i]
            f = fluxes[i](y[-1])*s
            total += f
            v[i] = f

        if multi_fluxes:
            v[nfluxes] = total
        return v

    # ODE Hessian - can be initialised outside to save time
    m = np.zeros((nfluxes+multi_fluxes, nfluxes+multi_fluxes)) \
        if m is None else m

    if method == "Radau":
        def jac_ivp(t, y):
            total = 0.
            for i in range(nfluxes):
                s = 1. if scaling is None else scaling[i]
                df = dfluxes[i](y[-1])*s
                total += df
                m[i, 0] = df

            if multi_fluxes:
                m[nfluxes, 0] = total
            return m
    else:
        jac_ivp = None

    y0 = [0.]*nfluxes+[s0] if multi_fluxes else [s0]

    res = solve_ivp(fun=fun_ivp,
                    t_span=[t0, t[-1]],
                    y0=y0,
                    method=method,
                    max_step=max_step,
                    jac=jac_ivp,
                    t_eval=t)

    # Function evaluation
    nev = res.nfev
    njac = res.njev if hasattr(res, "njev") else 0

    if len(res.t) == 0:
        return np.array([]), np.array([]), nev, njac
    else:
        return res.t, res.y.T.squeeze(), nev, njac


def numerical_model(fluxes, dfluxes, scalings, s0, timestep,
                    method="Radau"):
    nval = scalings.shape[0]
    sims = np.zeros(scalings.shape, dtype=np.float64)
    niter = np.zeros(nval, dtype=np.int32)
    s1 = np.zeros(nval, dtype=np.float64)
    nfluxes = len(fluxes)
    multi_fluxes = nfluxes > 1
    v = np.zeros(nfluxes+multi_fluxes)
    m = np.zeros((nfluxes+multi_fluxes, nfluxes+multi_fluxes))

    for t in range(nval):
        # integrate
        tn, send, nev, njac = integrate_numerical(fluxes,
                                                  dfluxes,
                                                  0, s0, [timestep],
                                                  scaling=scalings[t],
                                                  v=v, m=m,
                                                  method=method)
        s0 = send[-1]
        s1[t] = s0
        niter[t] = nev+njac
        sims[t] = send[:-1]

    return niter, s1, sims


# --- QUASOARE functions translated from C for slow implementation ---
def eta_fun(x, Delta):
    if Delta < 0.:
        return math.atan(x)
    else:
        if abs(x) < 1:
            return -math.atanh(x)
        elif abs(x) > 1:
            return -math.atanh(1./x)
        else:
            return np.inf if x < 0 else -np.inf


def omega_fun(x, Delta):
    return math.tan(x) if Delta < 0. else math.tanh(x)


def quad_fun(a, b, c, s):
    return (a*s+b)*s+c


def quad_grad(a, b, c, s):
    return 2*a*s+b


def quad_delta_t_max(a, b, c, Delta, qD, sbar, s0):
    if isnull(a):
        delta_tmax = np.inf
    else:
        tmp = a*(s0-sbar)
        if isnull(Delta):
            delta_tmax = np.inf if tmp <= 0. else 1./tmp
        elif Delta < 0:
            delta_tmax = (QUASOARE_PI/2.-eta_fun(tmp/qD, Delta))/qD
        else:
            delta_tmax = np.inf if tmp < qD else -eta_fun(tmp/qD, Delta)/qD

    return delta_tmax


def quad_forward(a, b, c, Delta, qD, sbar, t0, s0, t):
    if t < t0:
        return np.nan

    tau = t-t0
    dtmax = quad_delta_t_max(a, b, c, Delta, qD, sbar, s0)
    if tau < 0 or tau > dtmax:
        return np.nan

    if isequal(t0, t, QUASOARE_EPS, 0.):
        return s0

    if isnull(a) and isnull(b):
        s1 = s0+c*tau

    elif isnull(a) and notnull(b):
        s1 = -c/b+(s0+c/b)*math.exp(b*tau) \
                    if abs(b*tau) > QUASOARE_EPS \
                    else s0*(1+b*tau)+c*tau

    else:
        if isnull(Delta):
            s1 = sbar+(s0-sbar)/(1-a*tau*(s0-sbar))
        else:
            omega = omega_fun(qD*tau, Delta)
            signD = -1. if Delta < 0. else 1.
            s1 = sbar+(s0-sbar-signD*qD/a*omega)/(1-a/qD*(s0-sbar)*omega)

    return s1


def quad_inverse(a, b, c, Delta, qD, sbar, s0, s1):
    if isnull(a) and isnull(b):
        return (s1-s0)/c
    elif isnull(a) and notnull(b):
        return 1./b*math.log(abs((b*s1+c)/(b*s0+c)))
    else:
        if isnull(Delta):
            return (1./(s0-sbar)-1./(s1-sbar))/a
        else:
            return (eta_fun(a*(s1-sbar)/qD, Delta)
                    - eta_fun(a*(s0-sbar)/qD, Delta))/qD
    return np.nan


def quad_fluxes(aj_vector, bj_vector, cj_vector,
                Aj, Bj, Cj, Delta, qD, sbar,
                t0, t1, s0, s1, fluxes):
    tau = t1-t0
    tau2 = tau*tau
    tau3 = tau2*tau
    a = Aj
    b = Bj
    c = Cj

    if isnull(tau):
        return

    # Integrate S and S2
    if isnull(a) and isnull(b):
        integS = s0*tau+c*tau2/2
        integS2 = s0*s0*tau+s0*c*tau2+c*c*tau3/3.

    elif isnull(a) and notnull(b):
        integS = (s1-s0-Cj*tau)/Bj
        integS2 = ((s1**2-s0**2)/2-Cj*integS)/Bj

    elif notnull(a):
        if isnull(Delta):
            integS = sbar*tau-math.log(1-a*tau*(s0-sbar))/a
        else:
            omega = omega_fun(qD*tau, Delta)
            signD = -1 if Delta < 0. else 1.
            if qD*tau > 10. and Delta > 0:
                term1 = (math.log(2)-qD*tau)/Aj
            else:
                term1 = math.log(1-signD*omega*omega)/2./a
            term2 = -math.log(1-a*(s0-sbar)/qD*omega)/a
            integS = sbar*tau+term1+term2

    # increment fluxes
    if isnull(a):
        deltaf = aj_vector*integS2+bj_vector*integS+cj_vector*tau
    else:
        deltaf = aj_vector/a*(s1-s0)+(bj_vector-aj_vector*b/a)*integS\
                            + (cj_vector-aj_vector*c/a)*tau

    fluxes += deltaf


# Integrate reservoir equation over 1 time step and compute associated fluxes
def quad_integrate(alphas, scalings,
                   a_matrix_noscaling,
                   b_matrix_noscaling,
                   c_matrix_noscaling,
                   t0, s0, timestep, debug=False):

    nalphas = len(alphas)
    alpha_min = alphas[0]
    alpha_max = alphas[nalphas-1]

    # Initial interval
    jmax = nalphas-2
    if s0 < alpha_min:
        jalpha = -1
    elif s0 >= alpha_min and s0 < alpha_max:
        jalpha = np.sum(s0-alphas >= 0) - 1
    elif s0 == alpha_max:
        jalpha = jmax
    else:
        jalpha = jmax+1

    # Initialise iteration
    nfluxes = a_matrix_noscaling.shape[1]
    Aj, a_vect = 0., np.zeros(nfluxes)
    Bj, b_vect = 0., np.zeros(nfluxes)
    Cj, c_vect = 0., np.zeros(nfluxes)

    nit = 0
    niter_max = 2*nalphas
    t_final = t0+timestep
    t_start, t_end = t0, t0
    s_start, s_end = s0, s0
    fluxes = np.zeros(nfluxes)
    funval_prev = 0.

    if debug:
        print(f"\nNALPHAS = {nalphas} NFLUXES={a_matrix_noscaling.shape[1]}")
        print(f"Start t0={t0:5.5e} s0={s0:5.5e} j={jalpha}"
              + f" t_final={t_final:5.5e}")
        txt = " ".join([f"scl[{i}]={s:3.3e}" for i, s in enumerate(scalings)])
        print(f"scalings: {txt}")

    # Time loop
    while t_end < t_final*(1-QUASOARE_EPS) and nit < niter_max:
        nit += 1
        extrapolating_low = int(jalpha < 0)
        extrapolating_high = int(jalpha >= nalphas-1)
        extrapolating = int(extrapolating_low or extrapolating_high)

        # Get band limits
        alpha0 = -np.inf if extrapolating_low else alphas[jalpha]
        alpha1 = np.inf if extrapolating_high else alphas[jalpha+1]

        # Sum coefficients accross fluxes
        Aj = 0
        Bj = 0
        Cj = 0

        for i in range(nfluxes):
            if extrapolating_low:
                # Extrapolate approx as linear value
                a = a_matrix_noscaling[0, i]*scalings[i]
                b = b_matrix_noscaling[0, i]*scalings[i]
                c = c_matrix_noscaling[0, i]*scalings[i]

                grad = quad_grad(a, b, c, alpha_min)
                c = quad_fun(a, b, c, alpha_min)-grad*alpha_min
                b = grad
                a = 0

            elif extrapolating_high:
                # Extrapolate approx as linear value
                a = a_matrix_noscaling[nalphas-2, i]*scalings[i]
                b = b_matrix_noscaling[nalphas-2, i]*scalings[i]
                c = c_matrix_noscaling[nalphas-2, i]*scalings[i]

                grad = quad_grad(a, b, c, alpha_max)
                c = quad_fun(a, b, c, alpha_max)-grad*alpha_max
                b = grad
                a = 0

            else:
                # No extrapolation, use coefficients as is
                a = a_matrix_noscaling[jalpha, i]*scalings[i]
                b = b_matrix_noscaling[jalpha, i]*scalings[i]
                c = c_matrix_noscaling[jalpha, i]*scalings[i]

            a_vect[i] = a
            b_vect[i] = b
            c_vect[i] = c

            Aj += a
            Bj += b
            Cj += c

        # Round up coefficients
        Aj = 0. if abs(Aj) < QUASOARE_EPS else Aj
        Bj = 0. if abs(Bj) < QUASOARE_EPS else Bj

        # Discriminant
        Delta, qD, sbar = quad_constants(Aj, Bj, Cj)

        # Get derivative at beginning of time step
        funval = quad_fun(Aj, Bj, Cj, s_start)

        # Check continuity
        if nit > 1:
            if notequal(funval_prev, funval):
                errmess = f"continuity problem: prev({funval_prev:3.3e})"\
                        + f"!=new({funval:3.3e})"
                raise ValueError(errmess)

        # Try integrating up to the end of the time step
        if isnull(funval):
            s_end = s_start
        else:
            s_end = quad_forward(Aj, Bj, Cj, Delta, qD, sbar,
                                 t_start, s_start, t_final)

        # complete or move band if needed
        if s_end >= alpha0 and s_end <= alpha1:
            # .. s_end is within band => complete
            t_end = t_final
            jalpha_next = jalpha
        else:
            # .. find next band depending depending if f is decreasing
            #    or non-decreasing
            if extrapolating:
                # Ensure that s_end remains just inside interpolation range
                s_low = alpha_min+2*QUASOARE_EPS
                s_high = alpha_max-2*QUASOARE_EPS
                if funval < 0. and extrapolating_high and s_end < s_high:
                    s_end = s_high
                    jalpha_next = nalphas-2
                elif funval > 0. and extrapolating_low and s_end > s_low:
                    s_end = s_low
                    jalpha_next = 0
            else:
                if funval < 0.:
                    s_end = alpha0
                    jalpha_next = max(-1, jalpha-1)
                else:
                    s_end = alpha1
                    jalpha_next = min(nalphas-1, jalpha+1)

            # Increment time
            t_end = t_start+quad_inverse(Aj, Bj, Cj, Delta, qD, sbar,
                                         s_start, s_end)
            if t_end > t_final or abs(funval) < QUASOARE_EPS:
                t_end = t_final

        if debug:
            print(f"\n[{nit}] low={extrapolating_low}"
                  + f" high={extrapolating_high} "
                  + f"/ fun={funval:3.3e}"
                  + f"/ t={t_start:3.3e} > {t_end:3.3e}"
                  + f"/ j={jalpha} > {jalpha_next}"
                  + f" / s={s_start:3.3e} > {s_end:3.3e}")

        # Increment fluxes during the last interval
        quad_fluxes(a_vect, b_vect, c_vect,
                    Aj, Bj, Cj, Delta, qD, sbar,
                    t_start, t_end, s_start, s_end, fluxes)

        # Loop for next band
        funval_prev = quad_fun(Aj, Bj, Cj, s_end)

        if isnull(funval_prev):
            # f is null => we have reached steady state => complete
            t_end = t_final
            jalpha_next = jalpha
        else:
            t_start = t_end
            s_start = s_end
            jalpha = jalpha_next

    # Convergence problem
    if notequal(t_end, t_final, QUASOARE_ATOL, QUASOARE_RTOL)\
       and abs(funval) > QUASOARE_EPS:
        raise ValueError("No convergence")

    if debug:
        print(f"End integrate s1={s_end:5.5e} j={jalpha}")

    return nit, s_end, fluxes


def quad_model(alphas, scalings,
               a_matrix_noscaling,
               b_matrix_noscaling,
               c_matrix_noscaling, s0, timestep,
               errors="ignore", debug=False):
    assert errors in ERRORS
    nval = scalings.shape[0]
    fluxes = np.zeros(scalings.shape, dtype=np.float64)
    niter = np.zeros(nval, dtype=np.int32)
    s1 = np.zeros(nval, dtype=np.float64)
    t0 = 0.

    for t in range(nval):
        try:
            niter[t], s1[t], fluxes[t] = quad_integrate(alphas, scalings[t],
                                                        a_matrix_noscaling,
                                                        b_matrix_noscaling,
                                                        c_matrix_noscaling,
                                                        t0, s0, timestep,
                                                        debug=debug)
        except Exception as err:
            niter[t] = -1
            if errors == "raise":
                raise err
            elif errors == "warn":
                warnings.warn(str(err))

        s0 = s1[t]

    return niter, s1, fluxes
