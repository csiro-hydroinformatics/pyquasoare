import numpy as np

from pyquasoare import has_c_module

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


def quad_steady_flux_scalings(alphas, noscaling_coefs, flux_scalings,
                              out=None):
    """ Compute steady states using flux_scalings """
    nalphas = len(alphas)
    buff = np.empty(2 * nalphas + 2)
    if out is None:
        out = np.empty((flux_scalings.shape[0], 2 * nalphas + 2))
        out.fill(np.nan)

    ierr = c_pyquasoare.quad_steady_flux_scalings(alphas, noscaling_coefs,
                                                  flux_scalings, buff, out)
    if ierr > 0:
        errmsg = f"c_pyquasoare.quad_steady_flux_scalings returns {ierr}."
        raise ValueError(errmsg)

    return out
