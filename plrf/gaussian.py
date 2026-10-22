"""Exact second-moment matrices for Gaussian inputs and non-polynomial activations.

For a fixed sketch, (y_i, y_j) is centered Gaussian with variances s_i, s_j and covariance G_ij;
write rho_ij = G_ij / sqrt(s_i s_j).  Two exact methods:

1. Closed forms (Gaussian integrals):
     ReLU        E[f f'] = sqrt(s_i s_j) (sin t + (pi - t) cos t) / (2 pi),   cos t = rho,  mean sqrt(s / 2 pi)
     Heaviside   E[f f'] = (pi - t) / (2 pi),                                                mean 1/2
     y^2 e^{-y^2}  E[f f'] = D^{-1/2} (S11 S22 + 2 S12^2),  with M = I + 2 Sigma, D = det M,
                 S = Sigma M^{-1}:  S11 = (s_i (1 + 2 s_j) - 2 G^2) / D,  S22 likewise,  S12 = G / D;
                 mean s (1 + 2 s)^{-3/2}.
2. The Mehler (Hermite) series, for any f with E f(y)^2 < infinity:
     E[f(y_i) f(y_j)] = sum_k a_k(s_i) a_k(s_j) rho_ij^k,     a_k(s) = E[f(sqrt(s) Z) He_k(Z)] / sqrt(k!).
   Its partial sums are the decomposition of K_W by Hermite degree.  Off the diagonal |rho_ij| stays
   well below 1 (about 0.9 at most for the sketches used here), so truncating at degree L errs by at
   most max|rho|^L E f^2; the diagonal (rho = 1) is set to the exact E f(y_i)^2.

Matrix computations accept numpy arrays or torch tensors.
"""

from functools import lru_cache
from math import lgamma, log, pi, sqrt

import numpy as np

from plrf.activations import BUMP, HEAVISIDE, RELU


class _NS:
    """The few array operations needed, for numpy or torch."""

    def __init__(self, x):
        self.torch = type(x).__module__.startswith("torch")
        if self.torch:
            import torch
            self.m = torch
        else:
            self.m = np

    def clip(self, x, lo, hi):
        return self.m.clamp(x, lo, hi) if self.torch else np.clip(x, lo, hi)

    def outer(self, a, b):
        return self.m.outer(a, b)

    def diag(self, x):
        return self.m.diagonal(x).clone() if self.torch else np.diag(x).copy()

    def asarray(self, a, like):
        if self.torch:
            return self.m.as_tensor(a, dtype=like.dtype, device=like.device)
        return np.asarray(a)

    def fill_diag(self, K, x):
        if self.torch:
            K.diagonal().copy_(x)
        else:
            np.fill_diagonal(K, x)


def correlations(G):
    ns = _NS(G)
    s = ns.diag(G)
    r = ns.m.sqrt(s)
    rho = ns.clip(G / ns.outer(r, r), -1.0, 1.0)
    ns.fill_diag(rho, 1.0 + 0.0 * s)   # exactly 1: arccos amplifies a rounding error of 1e-16 to 1e-8
    return s, rho


# ---- closed forms ----

def closed_form(G, act):
    """(E[f(y_i) f(y_j)])_ij and the mean vector, or None if no closed form is implemented."""
    ns, m = _NS(G), _NS(G).m
    s, c = correlations(G)
    if act is RELU or act is HEAVISIDE:
        t = m.arccos(c)
        if act is RELU:
            r = m.sqrt(s)
            return ns.outer(r, r) * (m.sqrt(1.0 - c * c) + (pi - t) * c) / (2 * pi), r / sqrt(2 * pi)
        return (pi - t) / (2 * pi), 0.0 * s + 0.5
    if act is BUMP:
        si, sj = s[:, None], s[None, :]
        D = (1 + 2 * si) * (1 + 2 * sj) - 4 * G * G
        S11 = (si * (1 + 2 * sj) - 2 * G * G) / D
        S22 = (sj * (1 + 2 * si) - 2 * G * G) / D
        S12 = G / D
        return (S11 * S22 + 2 * S12 * S12) / m.sqrt(D), s * (1 + 2 * s) ** -1.5
    return None


# ---- Hermite coefficients ----

@lru_cache(maxsize=8)
def _gauss_hermite(n):
    """Nodes and weights for E over Z ~ N(0,1).  numpy's hermegauss overflows for n of several hundred.
    The nodes are the eigenvalues of the Jacobi matrix (off-diagonal sqrt(k)).  The weights come from
    the Christoffel formula 1 / sum_{k<n} h_k(z)^2 (h_k = He_k / sqrt(k!)), evaluated with rescaling,
    because eigenvector-based weights are only accurate to 1e-16 in absolute terms, and the outer
    weights (down to 1e-700) multiply Hermite polynomials of size up to 1e100 and more."""
    off = np.sqrt(np.arange(1, n, dtype=float))
    z = np.linalg.eigvalsh(np.diag(off, 1) + np.diag(off, -1))
    p0, p1 = np.ones(n), z.copy()
    S = 1.0 + p1**2
    logscale = np.zeros(n)
    for k in range(1, n - 1):
        p0, p1 = p1, (z * p1 - np.sqrt(k) * p0) / np.sqrt(k + 1)
        S += p1**2
        big = np.abs(p1) > 1e100
        if big.any():
            f = np.abs(p1[big])
            p0[big] /= f; p1[big] /= f; S[big] /= f**2
            logscale[big] += np.log(f)
    logw = -(np.log(S) + 2 * logscale)
    w = np.exp(logw)
    return z, w / w.sum()


def hermite_functions(z, L, scale=None):
    """(len(z), L+1) array of scale * He_k(z) / sqrt(k!), by the normalized three-term recurrence.
    With scale = sqrt(quadrature weight) the values stay bounded even at the outermost nodes."""
    h = np.zeros((len(z), L + 1))
    h[:, 0] = 1.0 if scale is None else scale
    if L >= 1:
        h[:, 1] = z * h[:, 0]
    for k in range(1, L):
        h[:, k + 1] = (z * h[:, k] - sqrt(k) * h[:, k - 1]) / sqrt(k + 1)
    return h


def _half_line_moments(L):
    """I_m = E[He_m(Z) 1{Z > 0}] / sqrt(m!) for m = 0..L+1:  I_0 = 1/2, I_m = He_{m-1}(0) phi(0) / sqrt(m!)."""
    out = np.zeros(L + 2)
    out[0] = 0.5
    phi0 = 1 / sqrt(2 * pi)
    for m in range(1, L + 2):
        n = m - 1
        if n % 2 == 0:
            # He_n(0) = (-1)^{n/2} (n-1)!!;  |He_n(0)| / sqrt(m!) computed in logs
            lg = (lgamma(n + 1) - lgamma(n // 2 + 1) - (n // 2) * log(2)) - 0.5 * lgamma(m + 1)
            out[m] = (-1) ** (n // 2) * phi0 * np.exp(lg)
    return out


def hermite_coefficients(act, s, L, n_quad=1000):
    """(len(s), L+1) array a_k(s_i) = E[f(sqrt(s_i) Z) He_k(Z)] / sqrt(k!).
    Exact for ReLU and Heaviside; Gauss-Hermite quadrature with n_quad nodes otherwise."""
    s = np.asarray(s, dtype=float)
    if act is HEAVISIDE or act is RELU:
        I = _half_line_moments(L)
        if act is HEAVISIDE:
            a = I[: L + 1]
        else:
            # z 1{z>0} He_k = 1{z>0} (He_{k+1} + k He_{k-1}); normalized: sqrt(k+1) I_{k+1} + sqrt(k) I_{k-1}
            k = np.arange(L + 1)
            a = np.sqrt(k + 1) * I[1: L + 2] + np.sqrt(k) * np.r_[0.0, I[: L]]
        scale = np.sqrt(s) if act is RELU else np.ones_like(s)
        return scale[:, None] * a[None, :]
    z, w = _gauss_hermite(n_quad)
    h = hermite_functions(z, L, np.sqrt(w)) * np.sqrt(w)[:, None]
    F = act(np.sqrt(s)[:, None] * z[None, :])
    return F @ h


def mean_square(act, s, n_quad=1000):
    """E f(sqrt(s_i) Z)^2 by quadrature (exact closed forms are used where they exist)."""
    z, w = _gauss_hermite(n_quad)
    return (act(np.sqrt(np.asarray(s))[:, None] * z[None, :]) ** 2) @ w


# ---- the Mehler series ----

def mehler(G, A, degrees, diag=None):
    """sum_{k in degrees} a_k a_k^T o rho^k, with A the (d, L+1) coefficient array.  If diag is given,
    the diagonal is set to it (the exact E f(y_i)^2)."""
    ns = _NS(G)
    _, rho = correlations(G)
    A = ns.asarray(A, G)
    K = 0.0 * G
    P = 1.0 + 0.0 * G
    degrees = sorted(degrees)
    for k in range(degrees[-1] + 1):
        if k:
            P = P * rho
        if k in degrees:
            K = K + ns.outer(A[:, k], A[:, k]) * P
    if diag is not None:
        ns.fill_diag(K, ns.asarray(diag, G))
    return K


def truncation_degree(G, tol=1e-14, lo=40, hi=2000):
    """Smallest L with max_{i != j} |rho_ij|^L < tol."""
    ns = _NS(G)
    _, rho = correlations(G)
    rho = ns.m.abs(rho)
    ns.fill_diag(rho, 0.0 * ns.diag(rho))
    r = float(rho.max())
    return int(min(hi, max(lo, np.ceil(np.log(tol) / np.log(r)))))


def second_moment_and_mean(G, act, L=None, n_quad=1000):
    """(K, mu): K = (1/d) E_x[f(y) f(y)^T] (uncentered) and mu = E_x[f(y)] for y ~ N(0, G).
    Closed form where available, otherwise the Mehler series truncated at degree L (chosen from
    max |rho_ij| if not given)."""
    d = G.shape[0]
    cf = closed_form(G, act)
    if cf is not None:
        return cf[0] / d, cf[1]
    ns = _NS(G)
    s = ns.diag(G)
    s_np = s.cpu().numpy() if ns.torch else s
    L = L or truncation_degree(G)
    A = hermite_coefficients(act, s_np, L, n_quad)
    K = mehler(G, A, range(L + 1), diag=mean_square(act, s_np, n_quad))
    return K / d, ns.asarray(A[:, 0], G)
