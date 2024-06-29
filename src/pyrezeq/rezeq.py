import math
import numpy as np

from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
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


def steady_state(nu, a, b, c):
    if isinstance(a, np.ndarray):
        n = len(a)
        steady = np.zeros((n, 2))
        ierr = c_pyrezeq.steady_state_vect(nu, a, b, c, steady)
    else:
        steady = np.zeros(2)
        ierr = c_pyrezeq.steady_state(nu, a, b, c, steady)
    return steady


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


def get_coefficients(fun, dfun, alphaj, alphajp1, nu, epsilon, ninterp=500):
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
    has_epsilon = not epsilon is None
    if has_epsilon:
        assert epsilon==-1 or (epsilon>=0 and epsilon<=1)

    # Basic constraints
    fa = lambda a, b, c, x: approx_fun(nu, a, b, c, x)
    X  = [[fa(1, 0, 0, x), fa(0, 1, 0, x), fa(0, 0, 1, x)]\
                        for x in [alphaj, alphajp1]]
    y = [fun(alphaj), fun(alphajp1)]

    if has_epsilon:
        # Epsilon is predefined - Additional constraint
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
    else:
        # Fit epsilon
        xx = np.linspace(alphaj, alphajp1, ninterp)
        yy = fun(xx)
        Xn = np.row_stack([X, [0, 0, 0]])
        yn = np.concatenate([y, [0]])

        def ofun_utils(theta):
            eps = expit(theta)
            x = (1-eps)*alphaj+eps*alphajp1
            Xn[2] = [fa(1, 0, 0, x), fa(0, 1, 0, x), fa(0, 0, 1, x)]
            yn[2] = fun(x)
            return x, Xn, yn

        def ofun(theta):
            x, Xn, yn = ofun_utils(theta)
            yn[2] = fun(x)
            a, b, c = np.linalg.solve(Xn, yn)
            yyhat = approx_fun(nu, a, b, c, xx)
            return np.abs(yyhat-yy).max()

        opt = minimize_scalar(ofun, [-5, 5], method="Bounded", bounds=[-5, 5])
        _, X, y = ofun_utils(opt.x)

    # Solution
    return np.linalg.solve(X, y)


def get_coefficients_matrix(funs, dfuns, alphas, nus, epsilons=None, ext=1e-3):
    """ Generate coefficient matrices for flux functions """
    nalphas = len(alphas)
    # Default
    # .. option to optimize?
    epsilons = 0.5*np.ones(nalphas-1) if epsilons is None else epsilons

    nus = np.ones(nalphas-1) if nus is None else nus
    if np.isscalar(nus):
        nus = nus*np.ones(nalphas-1)

    assert len(nus) == nalphas-1
    has_epsilons = not epsilons is None
    if has_epsilons:
        assert len(epsilons) == nalphas-1
    nfluxes = len(funs)

    # we add one row at the top end bottom for continuity extension
    a_matrix = np.zeros((nalphas+1, nfluxes))
    b_matrix = np.zeros((nalphas+1, nfluxes))
    c_matrix = np.zeros((nalphas+1, nfluxes))
    for j in range(nalphas-1):
        nu = nus[j]
        epsilon = epsilons[j] if has_epsilons else None
        alphaj, alphajp1 = alphas[[j, j+1]]

        for ifun, (f, df) in enumerate(zip(funs, dfuns)):
            a, b, c = get_coefficients(f, df, alphaj, alphajp1,\
                                                    nu, epsilon)
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



def increment_fluxes(scalings, nus, \
                        a_vector_noscaling, \
                        b_vector_noscaling, \
                        c_vector_noscaling, \
                        aoj, boj, coj, \
                        t0, t1, s0, s1, fluxes):

    ierr = c_pyrezeq.increment_fluxes(scalings, nus, \
                            a_vector_noscaling, \
                            b_vector_noscaling, \
                            c_vector_noscaling, \
                            aoj, boj, coj, \
                            t0, t1, s0, s1, fluxes)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.integrate returns {ierr}")


def steady_state_scalings(alphas, scalings, nus, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling):
    """ uses extended matrices = outputs from get_coefficients_matrix """
    nalphas = len(alphas)
    nval, nfluxes = scalings.shape

    for m in [a_matrix_noscaling, b_matrix_noscaling, c_matrix_noscaling]:
        assert len(m) == nalphas+1
        assert m.shape[1] == nfluxes

    steady = []
    for j in range(1, nalphas):
        nu = nus[j]
        a0 = scalings@a_matrix[j]
        b0 = scalings@b_matrix[j]
        c0 = scalings@c_matrix[j]
        s = steady_state(nu, a0, b0, c0)
        # TODO



def integrate(delta, alphas, scalings, nus, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling, \
                s0):
    fluxes = np.zeros(a_matrix_noscaling.shape[1], dtype=np.float64)
    s1 = np.zeros(1, dtype=np.float64)
    ierr = c_pyrezeq.integrate(delta, scalings, nus, \
                    a_matrix_noscaling, b_matrix_noscaling, \
                    c_matrix_noscaling, s0, s1, fluxes)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.integrate returns {ierr}")

    return u1[0], fluxes



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


def quadrouting(delta, theta, q0, s0, inflows, \
                    engine="C"):
    inflows = np.array(inflows).astype(np.float64)
    outflows = np.zeros_like(inflows)
    ierr = c_pyrezeq.quadrouting(delta, theta, q0, s0, inflows, outflows)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.quadrouting returns {ierr}")

    return outflows


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
