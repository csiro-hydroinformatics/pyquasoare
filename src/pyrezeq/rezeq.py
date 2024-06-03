import numpy as np

from scipy.optimize import minimize
from scipy.integrate import solve_ivp

from pyrezeq import has_c_module
if has_c_module():
    import c_pyrezeq

def check_alphas(alphas):
    assert len(alphas)>2, "Expected len(alphas)>2"
    errmess = "Expected strictly increasing alphas"
    assert np.all(np.diff(alphas)>0), errmess


def piecewise_linear_approximation(fun, alphas):
    """ Approximate a function with piecewise linear """
    check_alphas(alphas)
    yi = np.array([fun(u) for u in alphas])
    b = np.diff(yi)/np.diff(alphas)
    a = yi[:-1]-b*alphas[:-1]
    return np.column_stack([a, b])


#def get_piecewise_basis(x, alphas):
#    check_alphas(alphas)
#    nalphas = len(alphas)
#    X = np.zeros((len(x), nalphas))
#    for i in range(nalphas):
#        a1 = alphas[max(0, i-1)]
#        a2 = alphas[i]
#        a3 = alphas[min(i+1, nalphas-1)]
#
#        u = np.zeros_like(x)
#        kk = (x>=a1)&(x<a2)
#        u[kk] = (x[kk]-a1)/(a2-a1)
#        kk = (x>=a2)&(x<a3)
#        u[kk] = (x[kk]-a3)/(a2-a3)
#
#        X[:, i] = u
#
#    return X


def get_alphas(fun, alpha_min, alpha_max, nalphas, ninterp=1000):
    # First interpolation
    aa0 = np.linspace(alpha_min, alpha_max, ninterp)
    ff0 = np.array([fun(a) for a in aa0])

    # Minimization of sse
    def trans2raw(thetas):
        aa = np.cumsum(np.insert(np.exp(thetas), 0, 0))
        return alpha_min+(alpha_max-alpha_min)*(aa-aa[0])/(aa[-1]-aa[0])

    def objfun(theta):
        aa = trans2raw(theta)
        ff1 = [fun(a) for a in aa]
        ff0_int = np.interp(aa0, aa, ff1)
        sse = np.sum((ff0_int-ff0)**2)
        return sse

    a0 = np.linspace(alpha_min, alpha_max, nalphas)
    ini = np.log(np.diff(a0))
    opt = minimize(objfun, ini)
    alphas = trans2raw(opt.x)

    return alphas


def run_piecewise_approximation(x, alphas, coefs):
    check_alphas(alphas)
    y = np.zeros_like(x)
    nalphas = len(alphas)
    for i in range(nalphas-1):
        ulow = alphas[i]
        uhigh = alphas[i+1]
        ii = ((x>=ulow)&(x<uhigh))

        # Extrapolation
        if i==0:
            ii |= x<ulow
        if i==nalphas-2:
            ii |= x>=uhigh

        if ii.sum()>0:
            y[ii] = coefs[i, 0]+ coefs[i, 1]*x[ii]

    return y


def integrate_forward(t0, u0, a, b, t):
    return c_pyrezeq.integrate_forward(t0, u0, a, b, t)


def integrate_inverse(t0, u0, a, b, u):
    return c_pyrezeq.integrate_inverse(t0, u0, a, b, u)


def find_alpha(u0, alphas):
    alphas = np.array(alphas).astype(np.float64)
    return c_pyrezeq.find_alpha(u0, alphas)


def increment_fluxes(i_alpha, aoj, boj, t0, t1, u0, u1, scalings, \
                        a_matrix_noscaling, b_matrix_noscaling, \
                        fluxes):
    a_matrix_noscaling = np.array(a_matrix_noscaling).astype(np.float64)
    b_matrix_noscaling = np.array(b_matrix_noscaling).astype(np.float64)
    scalings = np.array(scalings).astype(np.float64)
    fluxes = np.array(fluxes).astype(np.float64)

    ierr = c_pyrezeq.increment_fluxes(i_alpha, aoj, boj, \
                        t0, t1, u0, u1, scalings, \
                        a_matrix_noscaling, b_matrix_noscaling, fluxes)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.integrate returns {ierr}")

    return fluxes


def integrate(delta, u0, alphas, scalings, \
                a_matrix_noscaling, b_matrix_noscaling):
    a_matrix_noscaling = np.array(a_matrix_noscaling).astype(np.float64)
    b_matrix_noscaling = np.array(b_matrix_noscaling).astype(np.float64)
    scalings = np.array(scalings).astype(np.float64)
    fluxes = np.zeros(a_matrix_noscaling.shape[1], dtype=np.float64)
    u1 = np.zeros(1, dtype=np.float64)

    ierr = c_pyrezeq.integrate(delta, u0, alphas, scalings, \
                    a_matrix_noscaling, b_matrix_noscaling, u1, fluxes)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.integrate returns {ierr}")

    return u1[0], fluxes


def run(delta, u0, alphas, scalings, \
                a_matrix_noscaling, b_matrix_noscaling):
    a_matrix_noscaling = np.array(a_matrix_noscaling).astype(np.float64)
    b_matrix_noscaling = np.array(b_matrix_noscaling).astype(np.float64)
    scalings = np.array(scalings).astype(np.float64)
    fluxes = np.zeros(scalings.shape, dtype=np.float64)
    u1 = np.zeros(scalings.shape[0], dtype=np.float64)

    ierr = c_pyrezeq.run(delta, u0, alphas, scalings, \
                    a_matrix_noscaling, b_matrix_noscaling, u1, fluxes)
    if ierr>0:
        raise ValueError(f"c_pyrezeq.run returns {ierr}")

    return u1, fluxes


def quadrouting(delta, theta, q0, s0, inflows):
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
