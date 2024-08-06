import math
import numpy as np

from pyrezeq import has_c_module
if has_c_module():
    import c_pyrezeq
    REZEQ_EPS = c_pyrezeq.get_eps()
    REZEQ_NFLUXES_MAX = c_pyrezeq.get_nfluxes_max()
else:
    raise ImportError("Cannot run rezeq without C code. Please compile C code.")

def all_scalar(*args):
    """ Check if all arguments are scalar """
    return all([np.isscalar(x) for x in args])


def get_vectors(*args, dtype=np.float64):
    """ Convert all arguments to vector of same length """
    v = [np.atleast_1d(x).astype(dtype) for x in args]
    nval = max([len(x) for x in v])
    ones = np.ones(nval)
    return [x[0]*ones if len(x)==1 else x for x in v]


def approx_fun(nu, a, b, c, s):
    """ Approximation exponential function f=a+b*exp(-nu.s)+c*exp(nu.s) """
    if all_scalar(a, b, c, s):
        return c_pyrezeq.approx_fun(nu, a, b, c, s)
    else:
        a, b, c, s, o = get_vectors(a, b, c, s, np.nan)
        ierr = c_pyrezeq.approx_fun_vect(nu, a, b, c, s, o)
        return o


def get_coefficients(fun, alphaj, alphajp1, nu, enforce_monotonous=False):
    """ Find approx coefficients for the interval [alphaj, alpjajp1]
        Fits the approx fun at x=a0, x=a1 and x=(a0+a1)/2.
        nu is the non-linearity coefficient.

        returns :
        a, b, c
        err(x=(3a0+a1)/4)
        err(x=(a0+3a1)/4)
    """
    assert alphajp1>alphaj
    assert nu>0

    # Points to be matched
    x0, x1 = alphaj, alphajp1
    xm = (x0+x1)/2
    e0, e1, em = math.exp(-nu*x0), math.exp(-nu*x1), math.exp(-nu*xm)
    f0, f1, fm = fun(x0), fun(x1), fun(xm)

    # Solution
    Uc, Vc = (f1-f0)/(1./e1-1./e0), -(e1-e0)/(1./e1-1./e0)
    Ua, Va = f0-Uc/e0, -Vc/e0-e0

    bplus = (fm-Ua-Uc/em)/(Va+em+Vc/em)

    # Potential correction of b to enforce the function
    # to be monotonous
    corrected = False
    if ((fm-f0)*(f1-fm)>-REZEQ_EPS or enforce_monotonous) and abs(f1-f0)>REZEQ_EPS:
        # f0<fm<f1 hence the function seems monotonous
        # Check function is monotone
        B1, B2 = (e0**2-Vc)/Uc, (e1**2-Vc)/Uc
        B1, B2 = min(B1, B2), max(B1, B2)

        invbp = 1/bplus
        if invbp>B1 and invbp<B2:
            b = 1/B1 if abs(invbp-B1)<abs(invbp-B2) else 1/B2
            corrected = True
        else:
            b = bplus
    else:
        b = bplus

    a = Ua+Va*b
    c = Uc+Vc*b

    return a, b, c, corrected


def get_coefficients_matrix(funs, alphas, nu):
    """ Generate coefficient matrices for flux functions """
    nalphas = len(alphas)
    nfluxes = len(funs)
    if nfluxes>REZEQ_NFLUXES_MAX:
        raise ValueError(f"Expected nfluxes<{REZEQ_NFLUXES_MAX}, "\
                            +f"got {nfluxes}.")

    # we add one row at the top end bottom for continuity extension
    a_matrix = np.zeros((nalphas-1, nfluxes))
    b_matrix = np.zeros((nalphas-1, nfluxes))
    c_matrix = np.zeros((nalphas-1, nfluxes))

    for j in range(nalphas-1):
        alphaj, alphajp1 = alphas[[j, j+1]]
        for ifun, f in enumerate(funs):
            a, b, c, _ = get_coefficients(f, alphaj, alphajp1, nu)
            a_matrix[j, ifun] = a
            b_matrix[j, ifun] = b
            c_matrix[j, ifun] = c

    return a_matrix, b_matrix, c_matrix


def approx_fun_from_matrix(alphas, nu, a_matrix, b_matrix, c_matrix, x):
    nalphas = len(alphas)
    nfluxes = a_matrix.shape[1]
    assert np.all(np.diff(alphas)>0)
    assert a_matrix.shape[0] == nalphas-1
    assert b_matrix.shape == a_matrix.shape
    assert c_matrix.shape == a_matrix.shape

    x = np.atleast_1d(x)
    outputs = np.nan*np.zeros((len(x), a_matrix.shape[1]))

    # Outside of alpha bounds
    idx = x<alphas[0]
    if idx.sum()>0:
        o = [approx_fun(nu, a, b, c, alphas[0]) for a, b, c \
                    in zip(a_matrix[0], b_matrix[0], c_matrix[0])]
        outputs[idx] = np.column_stack(o)

    idx = x>alphas[-1]
    if idx.sum()>0:
        o = [approx_fun(nu, a, b, c, alphas[-1]) for a, b, c \
                    in zip(a_matrix[-1], b_matrix[-1], c_matrix[-1])]
        outputs[idx] = np.column_stack(o)

    # Inside alpha bounds
    for j in range(nalphas-1):
        alphaj, alphajp1 = alphas[[j, j+1]]
        idx = (x>=alphaj-1e-10) & (x<=alphajp1+1e-10)
        if idx.sum()==0:
            continue

        for i in range(nfluxes):
            a = a_matrix[j, i]
            b = b_matrix[j, i]
            c = c_matrix[j, i]
            outputs[idx, i] = approx_fun(nu, a, b, c, x[idx])

    return outputs


def optimize_nu(funs, alphas, scalings_ref, nexplore=1000):
    """ Optimize nu and compute corresponding coefficients """

    a0, a1 = alphas.min(), alphas.max()
    theta_max = max(6., max(abs(math.asinh(a0)), abs(math.asinh(a1))))
    theta_min = -theta_max
    nalphas = len(alphas)
    assert len(funs) == len(scalings_ref)
    assert np.all(np.abs(scalings_ref)>10*REZEQ_EPS)

    # Error evaluation points for each band
    eps = np.array([1./4, 0.5, 3./4])[None, :]
    a = alphas[:, None]
    x_eval = a[:-1]*(1-eps)+a[1:]*eps

    # Flux sum function to be approximated including
    # reference scalings
    sfun = lambda x: sum([f(x)*scalings_ref[ifun] for ifun, f in enumerate(funs)])
    y_true = np.array([[sfun(x) for x in xe] for xe in x_eval])

    # Objective functions and derivatives
    def fobj(nu):
        of = 0
        for j in range(nalphas-1):
            # Process band j
            alphaj, alphajp1 = alphas[[j, j+1]]
            x = x_eval[j]
            # Get coefficients
            a, b, c, _ = get_coefficients(sfun, alphaj, alphajp1, nu)
            # Compute error
            yt = y_true[j]
            err = approx_fun(nu, a, b, c, x)-yt
            of += (err*err).mean()
        return of

    # Systematic exploration
    fobjexp = lambda x: fobj(math.exp(x))
    xa, xb = -5, 5
    ntry = 1000
    xx = np.linspace(xa, xb, ntry)
    ff = np.array([fobjexp(x) for x in xx])
    imin = np.argmin(ff)
    xa, xb = xx[max(0, imin-1)], xx[min(ntry-1, imin+1)]

    niter= ntry
    theta = xx[imin]
    fopt = fobjexp(theta)
    nu = math.exp(theta)

    ## Golden ratio minimization
    #dx = 10
    #niter = 0
    #niter_max = 500
    #tol = 1e-2
    #invphi = (math.sqrt(5)-1)/2
    #while abs(dx)>tol and niter<niter_max:
    #    dx = xb-xa
    #    xc = xb-dx*invphi
    #    xd = xa+dx*invphi
    #    if fobjexp(xc)<fobjexp(xd):
    #        xb = xd
    #    else:
    #        xa = xc

    #    niter += 1

    #theta = (xa+xb)/2
    #fopt = fobjexp(theta)
    #nu = math.exp(theta)

    # Get coefficient for unscaled functions
    amat, bmat, cmat = get_coefficients_matrix(funs, alphas, nu)

    # Check continuity
    f = []
    for j in range(nalphas-1):
        a0, a1 = alphas[[j, j+1]]
        a, b, c = amat[j].sum(), bmat[j].sum(), cmat[j].sum()
        f.append([\
            approx_fun(nu, a, b, c, a0), \
            approx_fun(nu, a, b, c, a1)
           ])
    f = np.array(f)
    assert np.allclose(f[1:, 0], f[:-1, 1])
    import pdb; pdb.set_trace()

    return nu, amat, bmat, cmat, niter, fopt

