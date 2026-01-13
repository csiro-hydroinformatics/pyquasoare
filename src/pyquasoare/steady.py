import numpy as np

from pyquasoare import has_c_module, approx

if has_c_module():
    import c_pyquasoare
else:
    raise ImportError("Cannot run quasoare without C code."
                      + " Please compile C code.")


def quad_steady(coefs, out=None):
    """ Compute steady state solution of QuaSoARe equation
        for a triplet of coefficients [a, b, c].
    """
    coefs = np.atleast_2d(coefs)
    if coefs.shape[1] == 1:
        coefs = coefs.T

    stdy = np.zeros((coefs.shape[0], 2)) if out is None else out
    c_pyquasoare.quad_steady(coefs, stdy)
    return stdy


def quad_steady_scalings(alphas, noscaling_coefs, scalings):
    """ Compute steady states using scalings """
    # Check inputs
    scalings = np.atleast_2d(scalings)
    if scalings.shape[0] == 1:
        scalings = scalings.T
    if scalings.ndim != 2:
        errmsg = f"Expected scalings of dim 2, got {scalings.ndim}."
        raise ValueError(errmsg)

    nalphas = len(alphas)
    if nalphas <= 2:
        errmsg = "Expected nalphas > 2."
        raise ValueError(errmsg)

    nval, nfluxes = scalings.shape
    if noscaling_coefs.shape[0] != nfluxes:
        errmsg = f"Expected {nfluxes} fluxes in noscaling_coefs,"\
                 + f" got {noscaling_coefs.shape[0]}."
        raise ValueError(errmsg)

    if noscaling_coefs.shape[1] != nalphas - 1:
        errmsg = f"Expected dim2 of length {nalphas - 1} in noscaling_coefs,"\
                 + f" got {noscaling_coefs.shape[1]}."
        raise ValueError(errmsg)

    if noscaling_coefs.shape[2] != 3:
        errmsg = "Expected dim3 of length 3 in noscaling_coefs,"\
                 + f" got {noscaling_coefs.shape[2]}."
        raise ValueError(errmsg)

    # Potentially a max of 2 x (nalphas+1) solutions
    # over nalphas-1 bands and 2 extrpolation if
    # there are 2 solutions for each band
    # (only possible for non-monotonous fuynctions)
    steady = np.zeros((nval, 2 * (nalphas + 1)))

    for j in range(-1, nalphas):
        if j >= 0 and j < nalphas-1:
            # General case
            coefs = scalings * noscaling_coefs[:, j].sum(axis=0)
            a0, a1 = alphas[[j, j+1]]

        elif j == -1:
            # Low extrapolation - Linear
            coefs = noscaling_coefs[:, 0].sum(axis=0)
            alpha_min = alphas[0]
            grad = approx.quad_grad(coefs, alpha_min)
            coefs[2] = approx.quad_fun(coefs, alpha_min) - grad * alpha_min
            coefs[1] = grad
            coefs[0] = 0.*grad
            coefs = scalings * coefs
            a0, a1 = -np.inf, alpha_min

        elif j == nalphas-1:
            # high extrapolation - Linear
            coefs = noscaling_coefs[:, -1].sum(axis=0)
            alpha_max = alphas[-1]
            grad = approx.quad_grad(coefs, alpha_max)
            coefs[2] = approx.quad_fun(coefs, alpha_max)-grad*alpha_max
            coefs[1] = grad
            coefs[0] = 0.*grad
            coefs = scalings * coefs
            a0, a1 = alpha_max, np.inf

        stdy = quad_steady(coefs)

        # Keep steady solution within band
        # .. ignore nan
        with np.errstate(invalid="ignore"):
            out_of_range = (stdy - a0) * (a1 - stdy) < 0
        stdy[out_of_range] = np.nan

        jc = j+1
        steady[:, 2 * jc : (2 * jc + 2)] = stdy

    # Reorder and remove nan columns
    steady = np.sort(steady, axis=1)
    hasvalid = np.any(~np.isnan(steady), axis=0)
    hasvalid = [0] if hasvalid.sum() == 0 else hasvalid
    steady = steady[:, hasvalid]

    return steady
