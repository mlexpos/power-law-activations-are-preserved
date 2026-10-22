"""Exact random-feature second-moment matrices for polynomial activations and inputs with
independent, non-Gaussian coordinates (no Monte Carlo).

Model: x = H^{1/2} xi with xi_k i.i.d. and standardized (plrf.inputs), y = W^T x.  The joint
cumulants of the pair (y_i, y_j) are sums over the independent coordinates:

    cum(a copies of y_i, b copies of y_j) = kappa_{a+b} sum_k H_k^{(a+b)/2} W_ki^a W_kj^b,

with kappa_2 = 1 and kappa_r the cumulants of xi.  So the bivariate cumulant generating function
of (y_i, y_j) is
    C(s, t) = sum_{a+b >= 2} C_ab s^a t^b / (a! b!),
where each C_ab is a d x d matrix (a vector when a or b is 0), and the moments are
    E[y_i^a y_j^b] = a! b! [s^a t^b] exp C(s, t).
For f(y) = sum_r c_r y^r of degree p only a, b <= p are needed, hence cumulants up to order 2p,
which exist for Student-t_nu only if nu > 2p.

For Gaussian xi (all kappa_r = 0 for r >= 3) this reduces to plrf.exact.second_moment.
"""

from math import factorial

import numpy as np

from plrf.activations import Activation, _WickPower


def _cumulant_series(W, H, p, cumulants):
    """{(a, b): C_ab / (a! b!)} for 2 <= a + b, a, b <= p.  Arrays broadcast to (d, d):
    (a, 0) -> (d, 1), (0, b) -> (1, d), otherwise (d, d)."""
    out = {}
    for a in range(p + 1):
        for b in range(p + 1):
            r = a + b
            kap = 1.0 if r == 2 else cumulants.get(r, 0.0)
            if r < 2 or kap == 0.0:
                continue
            h = H ** (r / 2)
            if b == 0:
                C = (h @ W**a)[:, None]
            elif a == 0:
                C = (h @ W**b)[None, :]
            else:
                C = (W**a).T @ (h[:, None] * W**b)
            out[(a, b)] = kap * C / (factorial(a) * factorial(b))
    return out


def _mul(A, B, p):
    """Product of truncated bivariate series (degrees <= p in each variable)."""
    out = {}
    for (a1, b1), x in A.items():
        for (a2, b2), y in B.items():
            a, b = a1 + a2, b1 + b2
            if a <= p and b <= p:
                out[(a, b)] = out[(a, b)] + x * y if (a, b) in out else x * y
    return out


def _moment_series(W, H, p, cumulants):
    """{(a, b): E[y_i^a y_j^b] / (a! b!)} for a, b <= p, as broadcastable arrays."""
    C = _cumulant_series(W, H, p, cumulants)
    M = {(0, 0): 1.0}
    term = {(0, 0): 1.0}
    for n in range(1, p + 1):   # every term of C has total degree >= 2, and we need total <= 2p
        term = {k: v / n for k, v in _mul(term, C, p).items()}
        for k, v in term.items():
            M[k] = M[k] + v if k in M else v
    return M


def second_moment_iid(W, H, activation, coords):
    """Exact (1/d) E_x[f(y) f(y)^T] for x = H^{1/2} xi with xi_k i.i.d. ~ coords (plrf.inputs);
    centered if activation.centered.  Polynomial activations only."""
    K, mu = second_moment_and_mean_iid(W, H, activation, coords)
    if activation.centered:
        K -= np.outer(mu, mu) / W.shape[1]
    return K


def second_moment_and_mean_iid(W, H, activation, coords):
    """(K, mu): the uncentered (1/d) E_x[f(y) f(y)^T] (activation.centered is ignored) and the mean
    vector E_x[f(y)], so that the covariance is K - mu mu^T / d."""
    if isinstance(activation, _WickPower) or not isinstance(activation, Activation):
        raise ValueError("second_moment_iid needs a polynomial activation with explicit coefficients")
    d = W.shape[1]
    p = activation.degree
    M = _moment_series(W, H, p, coords.cumulants(2 * p))
    c = activation.poly
    K = np.zeros((d, d))
    mu = np.zeros(d)
    for a in range(p + 1):
        if c[a] == 0:
            continue
        if (a, 0) in M:
            mu = mu + c[a] * factorial(a) * np.broadcast_to(M[(a, 0)], (d, 1))[:, 0]
        for b in range(p + 1):
            if c[b] != 0 and (a, b) in M:
                K += c[a] * c[b] * factorial(a) * factorial(b) * M[(a, b)]
    return K / d, mu
