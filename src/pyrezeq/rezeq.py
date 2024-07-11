import math
import numpy as np

from scipy.optimize import minimize_scalar, minimize
from scipy.special import expit, logit, softmax
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


def get_coefficients(fun, alphaj, alphajp1, nu, epsilon, ninterp=1000):
    """ Find approx coefficients for the interval [alphaj, alpjajp1]
        nu is the non-linearity coefficient.
        epsilon is the option controlling the third constraint:
        ]0, 1[ : use mid-point in ]alphaj, alphajp1[
    """
    assert alphajp1>alphaj
    has_epsilon = not epsilon is None
    if has_epsilon:
        assert epsilon>0 and epsilon<1

    # Basic constraints
    fa = lambda a, b, c, x: approx_fun(nu, a, b, c, x)
    X  = [[fa(1, 0, 0, x), fa(0, 1, 0, x), fa(0, 0, 1, x)]\
                        for x in [alphaj, alphajp1]]
    y = [fun(alphaj), fun(alphajp1)]

    if has_epsilon:
        x = (1-epsilon)*alphaj+epsilon*alphajp1
        X.append([fa(1, 0, 0, x), fa(0, 1, 0, x), fa(0, 0, 1, x)])
        y.append(fun(x))
    else:
        # Fit epsilon
        xx = np.linspace(alphaj, alphajp1, ninterp)
        yy = fun(xx)
        Xn = np.row_stack([X, [0, 0, 0]])
        yn = np.concatenate([y, [0]])

        def trans2true(theta):
            eps = expit(theta)
            x = (1-eps)*alphaj+eps*alphajp1
            Xn[2] = [fa(1, 0, 0, x), fa(0, 1, 0, x), fa(0, 0, 1, x)]
            yn[2] = fun(x)
            return x, Xn, yn

        def ofun(theta):
            x, Xn, yn = trans2true(theta)
            yn[2] = fun(x)
            a, b, c = np.linalg.solve(Xn, yn)
            yyhat = approx_fun(nu, a, b, c, xx)
            err = yyhat-yy
            return (err*err).sum()

        opt = minimize_scalar(ofun, [-5, 5], method="Bounded", bounds=[-5, 5])
        epsilon = expit(opt.x)
        _, X, y = trans2true(opt.x)

    # Solution
    return np.linalg.solve(X, y), epsilon


def get_coefficients_matrix(funs, alphas, nus=1, epsilons=None):
    """ Generate coefficient matrices for flux functions """
    nalphas = len(alphas)
    nfluxes = len(funs)

    # Default
    # .. option to optimize?
    nus = np.ones(nalphas-1) if nus is None else nus
    if np.isscalar(nus):
        nus = nus*np.ones(nalphas-1)

    assert len(nus) == nalphas-1
    has_epsilons = not epsilons is None
    if has_epsilons:
        if np.isscalar(epsilon):
            epsilons = epsilons*np.ones((nalphas-1, nfluxes))
        assert epsilons.shape == (nalphas-1, nfluxes)
    else:
        epsilons = 0.5*np.ones((nalphas-1, nfluxes))

    # we add one row at the top end bottom for continuity extension
    a_matrix = np.zeros((nalphas-1, nfluxes))
    b_matrix = np.zeros((nalphas-1, nfluxes))
    c_matrix = np.zeros((nalphas-1, nfluxes))
    for j in range(nalphas-1):
        nu = nus[j]
        alphaj, alphajp1 = alphas[[j, j+1]]

        for ifun, f in enumerate(funs):
            epsilon = epsilons[j] if has_epsilons else None
            (a, b, c), e = get_coefficients(f, alphaj, alphajp1,\
                                                    nu, epsilon)
            a_matrix[j, ifun] = a
            b_matrix[j, ifun] = b
            c_matrix[j, ifun] = c

            if not has_epsilons:
                epsilons[j, ifun] = e

    return nus, a_matrix, b_matrix, c_matrix, epsilons


def approx_fun_from_matrix(alphas, nus, a_matrix, b_matrix, c_matrix, s):
    nalphas = len(alphas)
    nfluxes = a_matrix.shape[1]
    if np.isscalar(nus):
        nus = nus*np.ones(nalphas-1)
    assert a_matrix.shape[0] == nalphas-1
    assert len(nus) == nalphas-1
    assert b_matrix.shape == a_matrix.shape
    assert c_matrix.shape == a_matrix.shape

    outputs = np.nan*np.zeros((len(s), a_matrix.shape[1]))

    # Outside of alpha bounds
    idx = s<alphas[0]
    if idx.sum()>0:
        o = [approx_fun(nus[0], a, b, c, alphas[0]) for a, b, c \
                    in zip(a_matrix[0], b_matrix[0], c_matrix[0])]
        outputs[idx] = np.column_stack(o)

    idx = s>alphas[-1]
    if idx.sum()>0:
        o = [approx_fun(nus[-1], a, b, c, alphas[-1]) for a, b, c \
                    in zip(a_matrix[-1], b_matrix[-1], c_matrix[-1])]
        outputs[idx] = np.column_stack(o)

    # Inside alpha bounds
    for j in range(nalphas-1):
        nu = nus[j]
        alphaj, alphajp1 = alphas[[j, j+1]]
        idx = (s>=alphaj-1e-10)&(s<=alphajp1+1e-10)
        if idx.sum()==0:
            continue

        for i in range(nfluxes):
            a = a_matrix[j, i]
            b = b_matrix[j, i]
            c = c_matrix[j, i]
            outputs[idx, i] = approx_fun(nu, a, b, c, s[idx])

    return outputs


def approx_error(funs, alphas, nus, a_matrix, b_matrix, c_matrix, \
                        errfun="max", ninterp=1000):
    """ Compute maximum approximation error and error on first difference """
    x = np.linspace(alphas[0], alphas[-1], ninterp)
    y = np.column_stack([f(x) for f in funs])
    yhat = approx_fun_from_matrix(alphas, nus, a_matrix, b_matrix, c_matrix, x)
    err = np.abs(yhat-y)
    f = getattr(np, errfun)
    return f(err, axis=0)


def get_coefficients_matrix_optimize(funs, alpha0, alpha1, nalphas, \
                                        nu0=0.1, nu1=10., nexplore=1000, \
                                        errfun="max"):
    """ Optimize alphas and nus. Caution: use low nalphas"""
    assert alpha0<alpha1
    assert nu0<nu1

    # sum function
    def sfun(x):
        y = np.zeros_like(x)
        for f in funs:
            y += f(x)
        return y

    # Optimization functions
    def trans2true(theta):
        u = softmax(theta[:nalphas-1])
        u = np.insert(np.cumsum(u), 0, 0)
        alphas = alpha0+(alpha1-alpha0)*u
        nus = nu0+(nu1-nu0)*expit(theta[nalphas-1:])
        return alphas, nus

    def ofun(theta):
        # avoids too small delta in alphas
        if np.any(np.abs(theta[:nalphas-1])>2):
            return np.inf
        alphas, nus = trans2true(theta)
        _, amat, bmat, cmat, _ = get_coefficients_matrix([sfun], alphas, nus)
        return approx_error([sfun], alphas, nus, amat, bmat, cmat, \
                                        errfun=errfun)

    omin = np.inf
    for i in range(nexplore):
        p = np.random.uniform(-1, 1, nalphas*2-2)
        p[:nalphas-1] *= 2
        p[nalphas-1:] *= 5
        o = ofun(p)
        if o<omin:
            ini = p
            omin = o

    opts = dict(maxiter=1000, maxfev=5000, xatol=1e-5, fatol=1e-5)
    opt = minimize(ofun, ini, method="Nelder-Mead", options=opts)
    alphas, nus = trans2true(opt.x)

    return alphas, nus


def steady_state_scalings(alphas, nus, scalings, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling):
    """ Compute steady states using scalings """
    # Check inputs
    nalphas = len(alphas)
    nval, nfluxes = scalings.shape
    for m in [a_matrix_noscaling, b_matrix_noscaling, c_matrix_noscaling]:
        assert len(m) == nalphas-1
        assert m.shape[1] == nfluxes

    steady = np.nan*np.zeros((nval, 2))
    for j in range(1, nalphas-1):
        a0 = scalings@a_matrix_noscaling[j]
        b0 = scalings@b_matrix_noscaling[j]
        c0 = scalings@c_matrix_noscaling[j]
        nu = np.ones_like(a0)*nus[j]
        s = steady_state(nu, a0, b0, c0)

        # eliminates steady states outside bounds
        s[s<alphas[j]] = np.nan
        s[s>alphas[j+1]] = np.nan
        isok = ~np.isnan(s)
        steady[isok] = s[isok]

    return np.array(steady)


def increment_fluxes(scalings, nu, \
                        a_vector_noscaling, \
                        b_vector_noscaling, \
                        c_vector_noscaling, \
                        aoj, boj, coj, \
                        t0, t1, s0, s1, fluxes):

    ierr = c_pyrezeq.increment_fluxes(scalings, nu, \
                            a_vector_noscaling, \
                            b_vector_noscaling, \
                            c_vector_noscaling, \
                            aoj, boj, coj, \
                            t0, t1, s0, s1, fluxes)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.integrate_flues returns {ierr}")


def integrate(delta, alphas, scalings, nus, \
                a_matrix_noscaling, \
                b_matrix_noscaling, \
                c_matrix_noscaling, \
                s0):
    # Initialise
    fluxes = np.zeros(a_matrix_noscaling.shape[1], dtype=np.float64)
    s1 = np.zeros(1, dtype=np.float64)

    # run
    ierr = c_pyrezeq.integrate(delta, alphas, scalings, nus, \
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



