"""Exact random-feature second-moment matrices for Gaussian inputs (no Monte Carlo).

Model: x ~ N(0, H) in R^v, sketch W in R^{v x d} with i.i.d. N(0,1) entries, features
f(W^T x) in R^d.  The object of the paper's main theorem is

    K_W = (1/d) E_x[ f(W^T x) f(W^T x)^T ].

For fixed W, y = W^T x ~ N(0, G) with G = W^T H W.  Writing f(y) = sum_k b_k(s) :y^k: with
s_i = G_ii (see plrf.activations) and using E[:y_i^k: :y_j^l:] = delta_{kl} k! G_ij^k,

    (K_W)_ij = (1/d) sum_k k! b_k(s_i) b_k(s_j) G_ij^k.

The k = 0 term is the rank-one mean term (1/d) E[f] E[f]^T; dropping it gives the covariance.
"""

from math import factorial

import numpy as np


def power_law_spectrum(v, alpha):
    """Diagonal of H with lambda_j(H) = j^{-alpha}, j = 1..v."""
    return np.arange(1, v + 1, dtype=float) ** (-alpha)


def sketch_gram(W, H):
    """G = W^T diag(H) W, the covariance of the pre-activations y = W^T x."""
    return W.T @ (H[:, None] * W)


def second_moment(G, activation):
    """Exact (1/d) E_x[f(y) f(y)^T] for y ~ N(0, G); centered if activation.centered."""
    d = G.shape[0]
    s = np.diag(G).copy()
    K = np.zeros_like(G)
    for k, b in activation.wick_coefficients(s).items():
        K += factorial(k) * np.outer(b, b) * (G**k if k else 1.0)
    return K / d


def mean_vector(G, activation):
    """E_x[f(y_i)] = b_0(s_i) (zero for odd monomials)."""
    s = np.diag(G).copy()
    b = activation.__class__(activation.name, activation.poly, False).wick_coefficients(s)
    return b.get(0, np.zeros_like(s))


def eigenvalues_desc(K):
    return np.linalg.eigvalsh(K)[::-1]


def relu_second_moment(G, centered=False):
    """Exact (1/d) E_x[relu(y) relu(y)^T] for y ~ N(0, G) (the arc-cosine kernel of degree 1; see
    plrf.gaussian, which also has Heaviside, y^2 exp(-y^2) and general activations)."""
    from plrf.activations import RELU
    from plrf.gaussian import closed_form
    K, mu = closed_form(G, RELU)
    if centered:
        K = K - np.outer(mu, mu)
    return K / G.shape[0]
