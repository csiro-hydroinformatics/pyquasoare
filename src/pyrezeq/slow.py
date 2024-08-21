import math
import numpy as np

from scipy.optimize import minimize
from scipy.integrate import solve_ivp

from pyrezeq.approx import REZEQ_EPS, REZEQ_PI
from pyrezeq.approx import isequal, notequal, isnull, notnull
from pyrezeq.integrate import quad_constants

def integrate_numerical(sumfun, dsumfun, fluxes, dfluxes, t0, s0, t, \
                            method="Radau", max_step=np.inf, \
                            fun_args=None):
    method = method.title()
    nfluxes = len(fluxes)
    assert len(dfluxes) == nfluxes

    v = np.zeros(nfluxes+1)
    def fun_ivp(t, y):
        v[0] = sumfun(y[0])
        for i in range(nfluxes):
            v[i+1] = fluxes[i](y[0])
        return v

    m = np.zeros((nfluxes+1, nfluxes+1))
    jac_ivp = None
    if method == "Radau":
        def jac_ivp(t, y):
            m[0, 0] = dsumfun(y[0])
            for i in range(nfluxes):
                m[i+1, 0] = dfluxes[i](y[0])
            return m

    res = solve_ivp(\
            fun=fun_ivp, \
            t_span=[t0, t[-1]], \
            y0=[s0]+[0.]*nfluxes, \
            method=method, \
            max_step=max_step, \
            jac=jac_ivp, \
            t_eval=t, \
            args=fun_args)

    # Function evaluation
    nev = res.nfev
    njac = res.njev if hasattr(res, "njev") else 0

    if len(res.t) == 0:
        return np.array([]), np.array([]), nev, njac
    else:
        return res.t, res.y.T.squeeze(), nev, njac


# --- REZEQ functions translated from C for slow implementation ---
def eta_fun(x, Delta):
    if Delta<0.:
        return math.atan(x)
    else:
        return -math.atanh(x) if abs(x)<1 else -math.atanh(1./x)


def omega_fun(x, Delta):
    return math.tan(x) if Delta<0. else math.tanh(x)


def quad_fun(a, b, c, s):
    return (a*s+b)*s+c

def quad_grad(a, b, c, s):
    return 2*a*s+b


def quad_delta_t_max(a, b, c, Delta, qD, sbar, s0):
    tmp = a*(s0-sbar)
    if isnull(a):
        delta_tmax = np.inf
    else:
        if isnull(Delta):
            delta_tmax = np.inf if tmp<=0 else 1./tmp
        elif Delta<0:
            delta_tmax = (REZEQ_PI/2.-eta_fun(tmp/qD, Delta))/qD
        else:
            delta_tmax = np.inf if tmp<qD else -eta_fun(tmp/qD, Delta)/qD

    return delta_tmax


def quad_forward(a, b, c, Delta, qD, sbar, t0, s0, t):
    if t<t0:
        return np.nan

    tau = t-t0
    dtmax = quad_delta_t_max(a, b, c, Delta, qD, sbar, s0)
    if tau<0 or tau>dtmax:
        return np.nan

    if isequal(t0, t, REZEQ_EPS, 0.):
        return s0

    if isnull(a) and isnull(b):
        s1 = s0+c*tau

    elif isnull(a) and notnull(b):
        s1 = -c/b+(s0+c/b)*math.exp(b*tau)

    else:
        if isnull(Delta):
            s1 = sbar+(s0-sbar)/(1-a*tau*(s0-sbar))
        else:
            omega = omega_fun(qD*tau, Delta)
            signD = -1. if Delta<0 else 1.
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
        elif Delta>0:
            eta = lambda x: math.atanh(x) if abs(x)<1 else math.atanh(1./x)
            return (eta(a*(s0-sbar)/qD)-eta(a*(s1-sbar)/qD))/qD
        else:
            return (math.atan(a*(s1-sbar)/qD)-math.atan(a*(s0-sbar)/qD))/qD

    return np.nan


def quad_fluxes(aj_vector, bj_vector, cj_vector, \
                        aoj, boj, coj, Delta, qD, sbar, \
                        t0, t1, s0, s1, fluxes):
    nfluxes = len(fluxes)
    tau = t1-t0
    tau2 = tau*tau
    tau3 = tau2*tau
    a = aoj
    b = boj
    c = coj

    # Integrate S and S2
    if isnull(a) and isnull(b):
        integS = s0*tau+c*tau2/2
        integS2 = s0*s0*tau+s0*c*tau2+c*c*tau3/3.

    elif isnull(a) and notnull(b):
        integS = (s1-s0-coj*tau)/boj;
        integS2 = ((s1**2-s0**2)/2-coj*integS)/boj;

    elif notnull(a):
        if isnull(Delta):
            integS = sbar*tau-math.log(1-a*tau*(s0-sbar))/a
        else:
            omega = omega_fun(qD*tau, Delta)
            signD = -1 if Delta<0 else 1
            if qD*tau>10 and Delta>0:
                term1 = (math.log(2)-qD*tau)/aoj
            else:
                term1 = math.log(1-signD*omega*omega)/2./a
            term2 = -math.log(1-a*(s0-sbar)/qD*omega)/a
            integS = sbar*tau+term1+term2

    # increment fluxes
    if isnull(a):
        fluxes += aj_vector*integS2+bj_vector*integS+cj_vector*tau
    else:
        fluxes += aj_vector/a*(s1-s0)+(bj_vector-aj_vector*b/a)*integS\
                            +(cj_vector-aj_vector*c/a)*tau


# Integrate reservoir equation over 1 time step and compute associated fluxes
def quad_integrate(alphas, scalings, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling, \
                t0, s0, timestep):

    nalphas = len(alphas)
    alpha_min=alphas[0]
    alpha_max=alphas[nalphas-1]
    debug = False

    if debug:
        print("")
        print("-"*50)
        print(f"nalphas = {nalphas}")
        txt = " ".join([f"scl[{i}]={s:0.3f}" for i, s in enumerate(scalings)])
        print(f"scalings: {txt}")

    # Initial interval
    jmin = 0
    jmax = nalphas-2
    if s0<alpha_min:
        jalpha = -1
    elif s0>=alpha_min and s0<alpha_max:
        jalpha = np.sum(s0-alphas>=0)-1
    elif s0 == alpha_max:
        jalpha = jmax
    else:
        jalpha = jmax+1

    # Initialise iteration
    nfluxes = a_matrix_noscaling.shape[1]
    aoj, aoj_prev, a_vect = 0., 0., np.zeros(nfluxes)
    boj, boj_prev, b_vect = 0., 0., np.zeros(nfluxes)
    coj, coj_prev, c_vect = 0., 0., np.zeros(nfluxes)

    nit = 0
    niter_max = 2*nalphas
    t_final = t0+timestep
    t_start, t_end = t0, t0
    s_start, s_end = s0, s0
    fluxes = np.zeros(nfluxes)

    # Time loop
    while t_final>t_end and nit<niter_max:
        nit += 1
        extrapolating_low = jalpha<0
        extrapolating_high = jalpha>=nalphas-1
        extrapolating = extrapolating_low or extrapolating_high

        # Get band limits
        alpha0 = -np.inf if extrapolating_low else alphas[jalpha]
        alpha1 = np.inf if extrapolating_high else alphas[jalpha+1]

        # Sum coefficients accross fluxes
        aoj = 0; boj = 0; coj = 0

        for i in range(nfluxes):
            if extrapolating_low:
                # Extrapolate approx as fixed value equal to f(alpha_min)
                a = a_matrix_noscaling[0, i]*scalings[i]
                b = b_matrix_noscaling[0, i]*scalings[i]
                c = c_matrix_noscaling[0, i]*scalings[i]

                grad = quad_grad(a, b, c, alpha_min)
                c = quad_fun(a, b, c, alpha_min)-grad*alpha_min
                b = grad
                a = 0

            elif extrapolating_high:
                # Extrapolate approx as fixed value equal to f(alpha_max)
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

            aoj += a
            boj += b
            coj += c

        # Discriminant
        Delta, qD, sbar = quad_constants(aoj, boj, coj)

        # Get derivative at beginning of time step
        funval = quad_fun(aoj, boj, coj, s_start)

        # Check continuity
        if nit>1:
            if notequal(funval_prev, funval):
                errmess = f"continuity problem: prev({funval_prev:3.3e})"\
                        +f"!=new({funval:3.3e})"
                raise ValueError(errmess)

        # Try integrating up to the end of the time step
        s_end = quad_forward(aoj, boj, coj, Delta, qD, sbar, \
                                    t_start, s_start, t_final)

        # complete or move band if needed
        if (s_end>=alpha0 and s_end<=alpha1 and not extrapolating)\
                        or isnull(funval):
            # .. s_end is within band => complete
            t_end = t_final
            jalpha_next = jalpha

            if isnull(funval):
                s_end = s_start
        else:
            # .. find next band depending depending if f is decreasing
            #    or non-decreasing
            if funval<0:
                s_end = alpha0
                jalpha_next = max(-1, jalpha-1)
            else:
                s_end = alpha1
                jalpha_next = min(nalphas-1, jalpha+1)

            # Increment time
            if extrapolating:
                # Extrapolation, we cut integration at t_final
                t_end = t_final
                # we also need to correct s_end that is infinite
                s_end = quad_forward(aoj, boj, coj, Delta, qD, sbar, \
                                                t_start, s_start, t_end)
                # Ensure that s_end remains just inside interpolation range
                if funval<0:
                    s_end = max(alpha_max-2*REZEQ_EPS, s_end)
                else:
                    s_end = min(alpha_min+2*REZEQ_EPS, s_end)
            else:
                # No extrapolation, we can reach s_end
                t_end = t_start+quad_inverse(aoj, boj, coj, Delta, qD, sbar, \
                                                                s_start, s_end)

        if debug:
            print(f"\n\n[{nit}] low={str(extrapolating_low)[0]}"\
                    +f" high={str(extrapolating_high)[0]} "\
                    +f"/ fun={funval:3.3e}"\
                    +f"/ t:={t_start:3.3e}>{t_end:3.3e}"\
                    +f"/ j:{jalpha}>{jalpha_next}"\
                    +f" / s:{s_start:3.3e}>{s_end:3.3e}")

        # Increment fluxes during the last interval
        quad_fluxes(a_vect, b_vect, c_vect, \
                    aoj, boj, coj, Delta, qD, sbar, \
                    t_start, t_end, s_start, s_end, fluxes)

        # Loop for next band
        funval_prev = quad_fun(aoj, boj, coj, s_end)
        #print(" "*4+f"f({nu:3.3e}, {aoj:3.3e}, {boj:3.3e}, "\
        #        +f"{coj:3.3e}, {s_end:3.3e})={funval_prev:3.3e}")
        t_start = t_end
        s_start = s_end
        jalpha = jalpha_next

    # Convergence problem
    if t_final>t_end:
        raise ValueError("No convergence")

    return nit, s_end, fluxes


