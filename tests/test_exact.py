"""The closed-form second-moment matrices agree with Monte Carlo over x.

Run:  python -m tests.test_exact
"""

import numpy as np

from plrf.activations import Activation, monomial
from plrf.exact import mean_vector, power_law_spectrum, second_moment, sketch_gram


def monte_carlo(W, H, f, m, centered, rng, batch=20000):
    d = W.shape[1]
    acc = np.zeros((d, d))
    mean = np.zeros(d)
    for _ in range(m // batch):
        Y = f((rng.standard_normal((batch, len(H))) * np.sqrt(H)) @ W)
        acc += Y.T @ Y
        mean += Y.sum(0)
    K = acc / m
    if centered:
        mu = mean / m
        K -= np.outer(mu, mu)
    return K / d


def test_against_monte_carlo(tol=0.03):
    rng = np.random.default_rng(1)
    v, d, m = 300, 60, 400_000
    H = power_law_spectrum(v, 1.31)
    W = rng.standard_normal((v, d))
    G = sketch_gram(W, H)
    cases = [monomial(1), monomial(2), monomial(3), monomial(2, centered=True),
             Activation("1 - y + 0.5 y^2 + 0.2 y^3", (1.0, -1.0, 0.5, 0.2))]
    for act in cases:
        K = second_moment(G, act)
        mc = monte_carlo(W, H, act, m, act.centered, rng)
        err = np.linalg.norm(mc - K) / np.linalg.norm(K)
        print(f"  {act.name:28s} relative Frobenius error vs Monte Carlo: {err:.4f}")
        assert err < tol, act.name
    # the mean of an even monomial is (p-1)!! s^{p/2}
    s = np.diag(G)
    assert np.allclose(mean_vector(G, monomial(2)), s)
    assert np.allclose(mean_vector(G, monomial(4)), 3 * s**2)
    assert np.allclose(mean_vector(G, monomial(3)), 0)


if __name__ == "__main__":
    test_against_monte_carlo()
    print("ok")
