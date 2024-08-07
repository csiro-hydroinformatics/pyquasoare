import math
import numpy as np

from scipy.optimize import minimize
from scipy.integrate import solve_ivp

from pyrezeq.approx import REZEQ_EPS, isequal

def integrate_forward_numerical(funs, dfuns, t0, s0, t, \
                            method="Radau", max_step=np.inf, \
                            fun_args=None):
    method = method.title()
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

    # Function evaluation
    nev = res.nfev
    njac = res.njev if hasattr(res, "njev") else 0

    return res.t, res.y.T.squeeze(), nev, njac


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

# --- REZEQ functions translated from C for slow implementation ---
def isnull(x):
    return 1 if abs(x)<REZEQ_EPS else 0

def notnull(x):
    return 1-isnull(x)

def ispos(x):
    return 1 if x>REZEQ_EPS else 0

def isneg(x):
    return 1 if x<-REZEQ_EPS else 0


def approx_fun(nu, a, b, c, s):
    f = a
    if notnull(b):
        f += b*math.exp(-nu*s)
    if notnull(c):
        f += c*math.exp(nu*s)
    return f

def integrate_delta_t_max(nu, a, b, c, s0):
    e0 = math.exp(-nu*s0)
    Delta = a*a-4*b*c

    if isnull(b) and isnull(c):
        delta_tmax = np.inf

    elif isnull(a) and isnull(b) and notnull(c):
        delta_tmax = np.inf if c<0 else e0/nu/c

    elif isnull(a) and notnull(b) and isnull(c):
        delta_tmax = np.inf if b>0 else -1/e0/nu/b

    elif notnull(a) and isnull(b) and notnull(c):
        delta_tmax = np.inf if c<0 or (c>0 and a<-c/e0) else math.log(1+a*e0/c)/nu/a

    elif notnull(a) and notnull(b) and isnull(c):
        delta_tmax = np.inf if b>0 or (b<0 and a>-b*e0) else -math.log(1+a/e0/b)/nu/a

    elif notnull(b) and notnull(c):
        sqD = math.sqrt(abs(Delta))
        lam0 = (2*b*e0+a)/sqD

        if isnull(Delta):
            delta_tmax = -2/(a+2*b*e0)/nu if a<-2*e0*b else np.inf
            delta_tmax = min(delta_tmax, 4*b*e0/(a+2*b*e0)/nu/a if a>-c/e0 else np.inf)

        elif isneg(Delta):
            delta_tmax = math.atan(-1./lam0)*2/nu/sqD if lam0<0 else np.inf
            tmp = math.atan((lam0*sqD-a)/(a*lam0+sqD))*2/nu/sqD
            tmp = tmp if tmp>0 else np.inf
            delta_tmax = min(min(delta_tmax, tmp), math.pi/nu/sqD)

        else:
            delta_tmax = math.atanh(-1./lam0)*2/nu/sqD if lam0<-1 else np.inf
            x = (lam0*sqD-a)/(a*lam0-sqD)
            tmp = math.atanh(x)*2/nu/sqD if abs(x)<1 else -np.inf
            tmp = tmp if tmp>0 else np.inf
            delta_tmax = min(delta_tmax, tmp)

    return delta_tmax if delta_tmax>=0 else np.nan


def integrate_forward(nu, a, b, c, t0, s0, t):
    e0 = math.exp(-nu*s0)
    Delta = a*a-4*b*c

    if t<t0:
        return np.nan

    dtmax = integrate_delta_t_max(nu, a, b, c, s0)
    if t-t0>dtmax:
        return np.nan

    if isequal(t0, t, REZEQ_EPS, 0.):
        return s0

    if isnull(b) and isnull(c):
        s1 = s0+a*(t-t0)

    elif isnull(a) and isnull(b) and notnull(c):
        s1 = s0-math.log(1-nu*c/e0*(t-t0))/nu

    elif isnull(a) and notnull(b) and isnull(c):
        s1 = s0+math.log(1+nu*b*e0*(t-t0))/nu

    elif notnull(a) and isnull(b) and notnull(c):
        s1 = s0+a*(t-t0)-math.log(1+c/a/e0*(1-math.exp(nu*a*(t-t0))))/nu

    elif notnull(a) and notnull(b) and isnull(c):
        s1 = s0+a*(t-t0)+math.log(1+b/a*e0*(1-math.exp(-nu*a*(t-t0))))/nu

    elif notnull(b) and notnull(c):
        ra2b = a/2/b
        if isnull(Delta):
            s1 = -math.log((e0+ra2b)/(1+(e0+ra2b)*nu*b*(t-t0))-ra2b)/nu

        else:
            sgn = -1 if isneg(Delta) else 1
            sqD = math.sqrt(sgn*Delta)
            omeg = math.tan(nu*sqD*(t-t0)/2) if isneg(Delta) \
                                else math.tanh(nu*sqD*(t-t0)/2)
            lam0 = (2*b*e0+a)/sqD
            s1 = -math.log((lam0+sgn*omeg)/(1+lam0*omeg)*sqD/2/b-ra2b)/nu

    return s1


def integrate_inverse(nu, a, b, c, s0, s1):
    e0 = math.exp(-nu*s0)
    e1 = math.exp(-nu*s1)
    Delta = a*a-4*b*c
    sgn = -1 if Delta<0 else 1

    if isnull(b) and isnull(c):
        return (s1-s0)/a

    elif isnull(a) and isnull(b) and notnull(c):
        return -e1/nu/c+e0/nu/c

    elif isnull(a) and notnull(b) and isnull(c):
        return 1./e1/nu/b-1./e0/nu/b

    elif notnull(a) and isnull(b) and notnull(c):
        return math.log((c+a*e0)/(c+a*e1))/nu/a

    elif notnull(a) and notnull(b) and isnull(c):
        return math.log((b+a/e1)/(b+a/e0))/nu/a

    elif notnull(b) and notnull(c):
        sqD = math.sqrt(sgn*Delta)
        lam0 = (2*b*e0+a)/sqD
        lam1 = (2*b*e1+a)/sqD

        if isnull(Delta):
            tau0 = 2./(a+2*b*e0)/nu
            tau1 = 2./(a+2*b*e1)/nu
            return tau1-tau0

        elif ispos(Delta):
            M = (1+lam1)*(1-lam0)/(1-lam1)/(1+lam0)
            return 1./sqD/nu*math.log(M) if M>0 else np.nan

        else:
            return -2./sqD/nu*(math.atan(lam1)-math.atan(lam0))

    return np.nan


def increment_fluxes(nu, aj_vector, bj_vector, cj_vector, \
                        aoj, boj, coj, \
                        t0, t1, s0, s1, fluxes):
    nfluxes = len(fluxes)
    dt = t1-t0
    e0 = math.exp(-nu*s0)
    expint = 0
    a = aoj
    b = boj
    c = coj
    Delta = aoj*aoj-4*boj*coj

    # Integrate exp(-nuS) if needed
    if notnull(b) or notnull(c):
        if isnull(a) and isnull(b) and notnull(c):
            expint = dt*e0-nu*c/2*dt*dt

        elif isnull(a) and notnull(b) and isnull(c):
            expint = dt/e0+nu*b/2*dt*dt

        elif notnull(a) and isnull(b) and notnull(c):
            expint = (e0+c/a)/nu/a*(1-math.exp(-nu*a*dt))-c/a*dt

        elif notnull(a) and notnull(b) and isnull(c):
            expint = -(1/e0+b/a)/nu/a*(1-math.exp(nu*a*dt))-b/a*dt

        elif notnull(b) and notnull(c):
            sqD = math.sqrt(abs(Delta))
            if isnull(Delta):
                expint = math.log(1+(e0+a/2/b)*nu*b*dt)/nu/b-a/2/b*dt

            else:
                lam0 = (2*b*e0+a)/sqD
                if ispos(Delta):
                    w = nu*sqD/2*dt
                    # Care with overflow
                    if w>100:
                        expint = (math.log((lam0+1)/2)+w)/nu/b-a/2/b*dt

                    else:
                        u1 = math.exp(w)
                        expint = math.log((lam0+1)*u1/2+(1-lam0)/u1/2)/nu/b-a/2/b*dt

                else:
                    u0 = math.atan(lam0)
                    u1 = u0-nu*sqD/2*dt
                    expint = math.log(math.cos(u1)/math.cos(u0))/nu/b-a/2/b*dt

    for i in range(nfluxes):
        aij = aj_vector[i]
        bij = bj_vector[i]
        cij = cj_vector[i]

        if isnull(b) and isnull(c):
            fluxes[i] += aij*dt
            if notnull(bij):
                fluxes[i] += -bij*e0/nu/a*(math.exp(-nu*a*dt)-1)
            if notnull(cij):
                fluxes[i] += cij/nu/a/e0*(math.exp(nu*a*dt)-1)
        else:
            if notnull(c):
                A = aij-cij*a/c
                B = bij-cij*b/c
                C = cij/c
                fluxes[i] += A*dt+B*expint+C*(s1-s0)
            else:
                A = aij-bij*a/b
                B = bij/b
                C = cij-bij*c/b
                fluxes[i] += A*dt+B*(s1-s0)+C*expint


# Integrate reservoir equation over 1 time step and compute associated fluxes
def integrate(alphas, scalings, nu, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling, \
                t0, s0, delta):

    nalphas = len(alphas)
    alpha_min=alphas[0]
    alpha_max=alphas[nalphas-1]

    #print("")
    #print("-"*50)
    #print(f"nalphas = {nalphas}")
    #txt = " ".join([f"scl[{i}]={s:0.3f}" for i, s in enumerate(scalings)])
    #print(f"scalings: {txt}")

    # Initial interval
    jmin = 0
    jmax = nalphas-2
    # jalpha < 0 or jalpha >= nalpha-1 => extrapolation
    jalpha = np.sum(s0-alphas>0)-1 if s0>=alpha_min else -1

    # Initialise iteration
    nfluxes = a_matrix_noscaling.shape[1]
    aoj, aoj_prev, a_vect = 0., 0., np.zeros(nfluxes)
    boj, boj_prev, b_vect = 0., 0., np.zeros(nfluxes)
    coj, coj_prev, c_vect = 0., 0., np.zeros(nfluxes)

    nit = 0
    t_final = t0+delta
    t_start, t_end = t0, t0
    s_start, s_end = s0, s0
    fluxes = np.zeros(nfluxes)

    # Time loop
    while ispos(t_final-t_end) and nit<nalphas:
        nit += 1
        extrapolating_low = jalpha<0
        extrapolating_high = jalpha>=nalphas-1
        extrapolating = extrapolating_low or extrapolating_high

        # Get band limits
        alpha0 = -np.inf if extrapolating_low else alphas[jalpha]
        alpha1 = np.inf if extrapolating_high else alphas[jalpha+1]

        # Store previous coefficients
        aoj_prev = aoj
        boj_prev = boj
        coj_prev = coj

        # Sum coefficients accross fluxes
        aoj = 0
        boj = 0
        coj = 0
        for i in range(nfluxes):
            if extrapolating_low:
                # Extrapolate approx as fixed value equal to f(alpha_min)
                a = a_matrix_noscaling[0, i]*scalings[i]
                b = b_matrix_noscaling[0, i]*scalings[i]
                c = c_matrix_noscaling[0, i]*scalings[i]
                a = approx_fun(nu, a, b, c, alpha_min)
                b, c = 0, 0
            elif extrapolating_high:
                # Extrapolate approx as fixed value equal to f(alpha_max)
                a = a_matrix_noscaling[nalphas-2, i]*scalings[i]
                b = b_matrix_noscaling[nalphas-2, i]*scalings[i]
                c = c_matrix_noscaling[nalphas-2, i]*scalings[i]
                a = approx_fun(nu, a, b, c, alpha_max)
                b, c = 0, 0
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

        # Get derivative at beginning of time step
        funval = approx_fun(nu, aoj, boj, coj, s_start)

        # Check continuity
        if nit>1:
            if approx.notequal(funval_prev, funval):
                raise ValueError("continuity problem")

        # Try integrating up to the end of the time step
        s_end = integrate_forward(nu, aoj, boj, coj, t_start, s_start, t_final)

        # complete or move band if needed
        if s_end>=alpha0 and s_end<=alpha1 and not extrapolating:
            # .. s_end is within band => complete
            t_end = t_final
            jalpha_next = jalpha
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
                s_end = integrate_forward(nu, aoj, boj, coj, t_start, s_start, t_end)
                # Ensure that s_end remains just inside interpolation range
                if funval<0:
                    s_end = max(alpha_max-2*REZEQ_EPS, s_end)
                else:
                    s_end = min(alpha_min+2*REZEQ_EPS, s_end)
            else:
                # No extrapolation, we can reach s_end
                t_end = t_start+integrate_inverse(nu, aoj, boj, coj, s_start, s_end)

        #print(f"[{nit}] low={extrapolating_low} high={extrapolating_high} "\
        #            +f"/ fun={funval:0.3f}"\
        #            +f"/ t:={t_start:0.3f}>{t_end:0.3f}"\
        #            +f"/ j:{jalpha}>{jalpha_next}"\
        #            +f" / s:{s_start:0.3f}>{s_end:0.3f}")

        # Increment fluxes during the last interval
        increment_fluxes(nu, a_vect, b_vect, c_vect, \
                    aoj, boj, coj, t_start, t_end, s_start, s_end, fluxes)

        # Loop for next band
        funval_prev = approx_fun(nu, aoj, boj, coj, s_end)
        t_start = t_end
        s_start = s_end
        jalpha = jalpha_next

    # Convergence problem
    if ispos(delta+t0-t_end):
        raise ValueError("No convergence")

    return nit, s_end, fluxes


