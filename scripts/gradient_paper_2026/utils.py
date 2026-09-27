import math
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from hydrodiy.stat import metrics

REPORT_PROBS = [0, 50, 90]

def smooth_nodes(x, npts=5, prc=10):
    x0, x1 = np.nanpercentile(x, [prc, 100 - prc])
    return np.linspace(x0, x1, npts)

def smooth_X(x, nodes):
    X = np.empty((len(x), len(nodes) + 2))
    X[:, 0] = 1.
    X[:, 1] = x
    for i, n in enumerate(nodes):
        dx = x - n
        X[:, i + 2] = np.abs(dx) * dx
    return X

def smooth_param(theta, out=None):
    return np.insert(theta, len(theta), -theta.sum())

def smooth_reg(x, y):
    nodes = smooth_nodes(x)
    iok = ~np.isnan(x) & ~np.isnan(y)
    X = smooth_X(x[iok], nodes)

    # Quantile reg objfun
    def ofun(theta):
        theta_ext = smooth_param(theta)
        yhat = X @ theta_ext
        return np.abs(yhat - y[iok]).sum()

    theta0 = np.zeros(X.shape[1] - 1)
    theta0[0] = np.nanmean(y) - np.nanmean(x)
    opt = minimize(ofun, theta0)
    theta = smooth_param(opt.x)

    xx = np.linspace(x[iok].min(), x[iok].max(), 500)
    XX = smooth_X(xx, nodes)
    yy = XX @ theta
    return xx, yy

def perf(qo, qs):
    eps = np.abs(qo.mean()) * 1e-2
    trans = {
        "raw": lambda x: x,
        "sqrt": lambda x: (np.sqrt(eps + np.abs(x)) - math.sqrt(eps)) * np.sign(x),
        "log": lambda x: (np.log(eps + np.abs(x)) - math.log(eps)) * np.sign(x),
        }
    perf = pd.DataFrame(index=trans.keys(), columns=["BIAS", "NSE"])
    for tn, t in trans.items():
        tqo, tqs = t(qo), t(qs)
        perf.loc[tn, "NSE"] = metrics.nse(tqo, tqs)
        perf.loc[tn, "BIAS"] = metrics.bias(tqo, tqs)
    return perf


def get_parameters():
    X0 = 1.
    X1 = 500.
    X2 = -0.5
    X3 = 50.
    X4 = 0.5
    return X0, X1, X2, X3, X4


def forwardP(x):
    return np.log(x + 1)


def backwardP(y):
    return np.maximum(np.exp(y) - 1, 0.)


def forwardE(x):
    return x


def backwardE(y):
    return np.maximum(y, 0.)


def random_scalings(sigma, nval=1, s0=1e-2):
    if nval == 1:
        return max(s0, 1 + np.random.normal(scale=sigma))
    else:
        return np.maximum(s0, 1 + np.random.normal(scale=sigma, size=nval))


def flux_isabove(afx, prob):
    thresh = np.percentile(afx, prob, axis=0)[None, :]
    return np.where(afx > thresh, 1., np.nan)


def flux_denom(fx, eps1=1e-4, eps2=1e-3):
    afx = np.abs(fx)
    means = np.mean(afx, axis=0)[None, :]
    maxs = np.max(afx, axis=0)[None, :]
    shift = eps1 + eps2 * means
    shift[maxs < eps1] = np.nan
    return afx + shift


def comp_dfluxes(fx_ref, dfx_ref, dfx_alt, eps1=1e-4, eps2=1e-3):
    num = np.abs(dfx_ref - dfx_alt)
    denom = flux_denom(fx_ref, eps1, eps2)
    err = num / denom

    emean = pd.Series(np.nan, index=REPORT_PROBS)
    stats = {}
    for prob in REPORT_PROBS:
        is_above = flux_isabove(afx, prob)[:, None, :, :]
        err_above = err * is_above

        with warnings.catch_warnings():
            # removes nan mean warning
            warnings.simplefilter("ignore")
            em = np.nanmean(err_above, axis=0)

        em[np.isnan(em)] = 0.
        emean[prob] = em.max()
        stats[prob] = em.squeeze()

    return emean, stats


