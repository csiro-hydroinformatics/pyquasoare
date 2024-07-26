import math
import numpy as np
from itertools import product as prod

from pyrezeq.rezeq import approx_fun_from_matrix, \
                get_coefficients_matrix, check_coefficient_matrix


def optimize_nu_and_epsilon(funs, alphas, nexplore=10000):
    """ Optimize alphas and nus. Caution: use low nalphas"""
    assert np.all(np.diff(alphas)>0)

    # sum of fluxes functions to be approximated
    xx = np.linspace(alphas.min(), alphas.max(), nexplore)
    yy = np.column_stack([f(xx) for f in funs]).sum(axis=1)

    # nu and eps values explored
    nus = np.arange(100)*0.1
    nus[0] = 0.01

    # .. start with 0.5 which is the most frequent option.
    # .. also test very small eps and eps close to 1 to match
    # .. boundary derivatives
    epsilons = [0.5, 1e-6, 1-1e-6]

    # Systematic exploration
    errmin = np.inf
    for nu, epsilon in prod(nus, epsilons):
        amat, bmat, cmat = get_coefficients_matrix(funs, alphas, nu, epsilon)

        has_problem = check_coefficient_matrix(funs, alphas, nu, \
                                                        amat, bmat, cmat)
        # the fit has problem, skip this config
        if has_problem.sum()>0:
            continue

        # The fit has no problem, calculate approximation error
        yy_approx = approx_fun_from_matrix(alphas, nu, amat, bmat, cmat, xx)
        yy_approx = yy_approx.sum(axis=1)
        err = math.sqrt(((yy_approx-yy)**2).mean())
        if err<errmin:
            nu_opt = nu
            epsilon_opt = epsilon
            amat_opt, bmat_opt, cmat_opt = amat, bmat, cmat
            errmin = err

    if np.isinf(errmin):
        raise ValueError("Cannot identify optimal nu and eps due to"\
                +" systematic interpolation issues. See"\
                +" rezeq.check_coefficient_matrix.")

    return nu_opt, epsilon_opt, amat_opt, bmat_opt, cmat_opt


