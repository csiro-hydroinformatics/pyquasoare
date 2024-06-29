import math
import numpy as np

from scipy.optimize import minimize
from scipy.integrate import solve_ivp

from pyrezeq import has_c_module
if has_c_module():
    import c_pyrezeq
    REZEQ_EPS = c_pyrezeq.get_eps()
else:
    REZEQ_EPS = 1e-10

def check_alphas(alphas):
    assert len(alphas)>2, "Expected len(alphas)>2"
    errmess = "Expected strictly increasing alphas"
    assert np.all(np.diff(alphas)>0), errmess


def approx_fun(nu, a, b, c, s):
    if isinstance(s, np.ndarray):
        ds = np.zeros_like(s)
        ierr = c_pyrezeq.approx_fun_vect(nu, a, b, c, s, ds)
        return ds
    else:
        return c_pyrezeq.approx_fun(nu, a, b, c, s)


def approx_jac(nu, a, b, c, s):
    if isinstance(s, np.ndarray):
        ds = np.zeros_like(s)
        ierr = c_pyrezeq.approx_jac_vect(nu, a, b, c, s, ds)
        return ds
    else:
        return c_pyrezeq.approx_jac(nu, a, b, c, s)


def integrate_forward(nu, a, b, c, t0, s0, t):
    if isinstance(t, np.ndarray):
        s = np.zeros_like(t)
        ierr = c_pyrezeq.integrate_forward_vect(nu, a, b, c, t0, s0, t, s)
        return s
    else:
        return c_pyrezeq.integrate_forward(nu, a, b, c, t0, s0, t)


def integrate_forward_numerical(fun, dfun, t0, s0, t, \
                            method="Radau", max_step=np.inf):
    v = np.zeros(1)
    def f(t, y):
        v[0] = fun(y[0])
        return v

    m = np.zeros((1, 1))
    jac = None
    if method == "Radau":
        def jac(t, y):
            m[0, 0] = dfun(y[0])
            return m

    res = solve_ivp(\
            fun=f, \
            t_span=[t0, t[-1]], \
            y0=[s0], \
            method=method, \
            max_step=max_step, \
            jac=jac, \
            t_eval=t)

    return res.t, res.y[0]



def integrate_delta_t_max(nu, a, b, c, s0):
    return c_pyrezeq.integrate_delta_t_max(nu, a, b, c, s0)


def integrate_inverse(nu, a, b, c, s0, s1):
    if isinstance(s1, np.ndarray):
        t = np.zeros_like(s1)
        ierr = c_pyrezeq.integrate_inverse_vect(nu, a, b, c, s0, s1, t)
        return t
    else:
        return c_pyrezeq.integrate_inverse(nu, a, b, c, s0, s1)


def find_alpha(alphas, u0):
    return c_pyrezeq.find_alpha(alphas, u0)


def get_coefficients(fun, dfun, alphaj, alphajp1, nu, epsilon):
    """ Find approx coefficients for the interval [alphaj, alpjajp1]
        nu is the non-linearity coefficient.
        epsilon is the option controlling the third constraint:
        -1 : set linearity constraint b=-c
        0 : use derivative in alphaj
        1 : use derivative in alphajp1
        ]0, 1[ : use mid-point in ]alphaj, alphajp1[
        ]
    """
    assert alphajp1>alphaj
    assert epsilon==-1 or (epsilon>=0 and epsilon<=1)

    # Basic constraints
    fa = lambda a, b, c, x: approx_fun(nu, a, b, c, x)
    X  = [[fa(1, 0, 0, x), fa(0, 1, 0, x), fa(0, 0, 1, x)]\
                        for x in [alphaj, alphajp1]]
    y = [fun(alphaj), fun(alphajp1)]

    # Additional constraint
    if epsilon==-1:
        # Equality b=-c
        X.append([0, 1, 1])
        y.append(0)
    elif epsilon in [0, 1]:
        # Derivative
        x = alphaj if epsilon==0 else alphajp1
        ja = lambda a, b, c, x: approx_jac(nu, a, b, c, x)
        X.append([ja(1, 0, 0, x), ja(0, 1, 0, x), ja(0, 0, 1, x)])
        y.append(dfun(x))
    else:
        # Mid-point
        x = (1-epsilon)*alphaj+epsilon*alphajp1
        X.append([fa(1, 0, 0, x), fa(0, 1, 0, x), fa(0, 0, 1, x)])
        y.append(fun(x))

    # Solution
    return np.linalg.solve(X, y)


def get_coefficients_matrix(funs, dfuns, alphas, nus, epsilons, ext=1e-3):
    """ Generate coefficient matrices for flux functions """
    nalphas = len(alphas)

    # Default
    # .. option to optimize?
    epsilons = 0.5*np.ones(nalphas-1) if epsilons is None else epsilons
    nus = np.ones(nalphas-1) if nus is None else nus

    assert len(nus) == nalphas-1
    assert len(epsilons) == nalphas-1
    nfuns = len(funs)

    # we add one row at the top end bottom for continuity extension
    a_matrix = np.zeros((nalphas+1, nfuns))
    b_matrix = np.zeros((nalphas+1, nfuns))
    c_matrix = np.zeros((nalphas+1, nfuns))

    for j in range(nalphas-1):
        nu = nus[j]
        epsilon = epsilons[j]
        alphaj, alphajp1 = alphas[[j, j+1]]

        for ifun, (f, df) in enumerate(zip(funs, dfuns)):
            a, b, c = get_coefficients(f, df, alphaj, alphajp1, nu, epsilon)
            a_matrix[j+1, ifun] = a
            b_matrix[j+1, ifun] = b
            c_matrix[j+1, ifun] = c

    # Add fixed derivative extension
    if ext>0:
        alpha0, alpha1 = alphas[[0, -1]]
        alphas_ext = np.concatenate([[alpha0-ext], alphas, [alpha1+ext]])
        nus_ext = np.concatenate([[nus[0]], nus, [nus[-1]]])
        for ifun, f in enumerate(funs):
            a_matrix[0, ifun] = f(alpha0)
            a_matrix[-1, ifun] = f(alpha1)

    return nus_ext, alphas_ext, a_matrix, b_matrix, c_matrix


def approx_fun_from_matrix(alphas, nus, a_matrix, b_matrix, c_matrix, s):
    nalphas = len(alphas)
    nfluxes = a_matrix.shape[1]
    assert a_matrix.shape[0] == nalphas-1
    assert len(nus) == nalphas-1
    assert b_matrix.shape == a_matrix.shape
    assert c_matrix.shape == a_matrix.shape

    outputs = np.nan*np.zeros((len(s), a_matrix.shape[1]))
    for j in range(nalphas-1):
        nu = nus[j]
        alphaj, alphajp1 = alphas[[j, j+1]]

        if j==0:
            idx = s<alphajp1
        elif j==nalphas-2:
            idx = s>=alphaj
        else:
            idx = (s>=alphaj)&(s<alphajp1)

        if idx.sum()==0:
            continue

        for i in range(nfluxes):
            a = a_matrix[j, i]
            b = b_matrix[j, i]
            c = c_matrix[j, i]
            outputs[idx, i] = approx_fun(nu, a, b, c, s[idx])

    return outputs



def increment_fluxes(scalings, \
                        a_vector_noscaling, \
                        b_vector_noscaling, \
                        c_vector_noscaling, \
                        aoj, boj, coj, \
                        t0, t1, s0, s1, fluxes):

    ierr = c_pyrezeq.increment_fluxes(scalings, \
                            a_vector_noscaling, \
                            b_vector_noscaling, \
                            c_vector_noscaling, \
                            aoj, boj, coj, \
                            t0, t1, s0, s1, fluxes)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.integrate returns {ierr}")


def integrate(delta, alphas, scalings, nu_vector, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling, \
                s0):
    fluxes = np.zeros(a_matrix_noscaling.shape[1], dtype=np.float64)
    s1 = np.zeros(1, dtype=np.float64)
    ierr = c_pyrezeq.integrate(delta, scalings, nu_vector, \
                    a_matrix_noscaling, b_matrix_noscaling, \
                    c_matrix_noscaling, s0, s1, fluxes)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.integrate returns {ierr}")

    return u1[0], fluxes


def integrate_python(delta, u0, alphas, scalings, \
                a_matrix_noscaling, b_matrix_noscaling):
    raise ValueError("TODO")
    # Dimensions
    nalphas = len(alphas)
    nfluxes = a_matrix_noscaling.shape[1]

    # Initialise
    aoj=0.
    boj=0.
    du1=0
    du2=0
    jalpha = find_alpha(u0, alphas)
    t0 = 0
    niter = 0
    aoj_prev = 0
    boj_prev = 0
    fluxes = np.zeros(nfluxes)

    # Time loop
    while t0<delta-1e-10 and niter<nalphas:
        niter += 1;

        # Store previous coefficients
        aoj_prev = aoj
        boj_prev = boj

        # Sum coefficients accross fluxes */
        aoj = 0
        boj = 0
        for j in range(nfluxes):
            aoj += a_matrix_noscaling[jalpha, j]*scalings[j]
            boj += b_matrix_noscaling[jalpha, j]*scalings[j]

        if np.isnan(aoj) or np.isnan(boj):
            return np.nan, np.nan*fluxes

        # Check continuity
        if niter>1:
            du1 = aoj_prev+boj_prev*u0
            du2 = aoj+boj*u0
            if abs(du1-du2)>1e-10:
                return np.nan, np.nan*fluxes

        # Get band limits
        ulow = alphas[jalpha]
        uhigh = alphas[jalpha+1]

        # integrate ODE up to the end of the time step
        u1 = integrate_forward(t0, u0, aoj, boj, delta)

        # Check if integration stays in the band or
        # if we are below lowest alphas or above highest alpha
        # In these cases, complete integration straight away.
        if u1>=ulow and u1<=uhigh:
            increment_fluxes(jalpha, aoj, boj, t0, delta, u0, u1, scalings, \
                        a_matrix_noscaling, b_matrix_noscaling, \
                        fluxes)
            t0 = delta
            u0 = u1

        else:
            if (jalpha==0 and u1<ulow) or (jalpha==nalphas-2 and u1>uhigh):
                # We are on the fringe of the alphas domain
                jalpha_next = jalpha
                t1 = delta

            else:
                # If not, decrease or increase parameter band
                # depending on increasing or decreasing nature
                # of ODE solution */
                if u1<=ulow:
                    jalpha_next = jalpha-1
                    u1 = ulow
                else:
                    jalpha_next = jalpha+1
                    u1 = uhigh

                # Find time where we move to the next band
                t1 = integrate_inverse(t0, u0, aoj, boj, u1)

            # Increment variables
            increment_fluxes(jalpha, aoj, boj, t0, t1, u0, u1, scalings, \
                        a_matrix_noscaling, b_matrix_noscaling, \
                        fluxes)
            t0 = t1
            u0 = u1
            jalpha = jalpha_next

    # Convergence problem
    if t0<delta-1e-10:
        return np.nan, np.nan*fluxes

    return u1, fluxes



def run(delta, alphas, scalings, \
                nu_vector, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling, \
                s0):
    fluxes = np.zeros(scalings.shape, dtype=np.float64)
    s1 = np.zeros(scalings.shape[0], dtype=np.float64)

    ierr = c_pyrezeq.run(delta, alphas, scalings, \
                    nu_vector, \
                    a_matrix_noscaling, \
                    b_matrix_noscaling, \
                    c_matrix_noscaling, \
                    s0, s1, fluxes)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.run returns {ierr}")

    return u1, fluxes



def run_python(delta, u0, alphas, scalings, \
                a_matrix_noscaling, b_matrix_noscaling):
    raise ValueError("TODO")
    fluxes = np.zeros(scalings.shape, dtype=np.float64)
    u1 = np.zeros(scalings.shape[0], dtype=np.float64)
    nval = len(scalings)

    for t in range(nval):
        u1[t], fluxes[t] = integrate_python(delta, u0, \
                            alphas, \
                            scalings[t], \
                            a_matrix_noscaling, \
                            b_matrix_noscaling)
        # Loop initial state
        u0 = u1[t]

    return u1, fluxes


def quadrouting(delta, theta, q0, s0, inflows, \
                    engine="C"):
    inflows = np.array(inflows).astype(np.float64)
    outflows = np.zeros_like(inflows)
    ierr = c_pyrezeq.quadrouting(delta, theta, q0, s0, inflows, outflows)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.quadrouting returns {ierr}")

    return outflows



def numrouting(delta, theta, q0, s0, inflows, nu, \
                        method="RK45", \
                        max_step_frac=0.1):
    def f(t, y, qi):
        return qi-q0*(y/theta)**nu

    def jac(t, y, qi):
        return np.array([-q0*nu*(y/theta)**(nu-1)])

    nval = len(inflows)
    fluxes = []
    store = []
    for t in range(nval):
        qi = inflows.iloc[t]
        res = solve_ivp(\
                fun=f, \
                t_span=[0, delta], \
                y0=[s0], \
                method=method, \
                args=(qi, ), \
                max_step=delta*max_step_frac, \
                jac=jac)
        s1 = res.y[0][-1]
        fluxes.append((s0-s1)/delta+qi)
        store.append(s1)

        # Loop
        s0 = s1

    return np.array(store), np.array(fluxes)
