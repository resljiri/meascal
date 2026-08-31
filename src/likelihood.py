from __future__ import annotations
import numpy as np


def log1mexp(q: np.ndarray) -> np.ndarray:
    q = np.minimum(q, -1e-14)
    out = np.empty_like(q)
    m = q < -np.log(2)
    out[m] = np.log1p(-np.exp(q[m]))
    out[~m] = np.log(-np.expm1(q[~m]))
    return out


def interval_weibull_loglik(L, R, eta, k, sentinel=99999.0):
    L = np.asarray(L, float)
    R = np.asarray(R, float)
    eta = np.asarray(eta, float)
    k = np.asarray(k, float)
    rc = R >= sentinel
    logSL = -(np.maximum(L, 0) / eta) ** k
    ll = np.empty(len(L))
    ll[rc] = logSL[rc]
    ev = ~rc
    if ev.any():
        logSR = -(R[ev] / eta[ev]) ** k[ev]
        ll[ev] = logSL[ev] + log1mexp(logSR - logSL[ev])
    return ll
