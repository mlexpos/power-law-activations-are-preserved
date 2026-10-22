"""Checks of plrf.gaussian (Gaussian inputs, non-polynomial activations).

1. The Mehler series reproduces the closed forms for ReLU, Heaviside and y^2 exp(-y^2), and the
   polynomial formula of plrf.exact for y^2 and y^3, at both input scales used in Experiment 3.
2. The closed forms and the Mehler series for tanh agree with Monte Carlo.
3. The Hermite coefficients of tanh are converged in the number of quadrature nodes.
4. numpy and torch give the same matrices.

Run:  python -m tests.test_gaussian
"""

import numpy as np

from plrf.activations import BUMP, HEAVISIDE, RELU, TANH, monomial
from plrf.exact import power_law_spectrum, second_moment, sketch_gram
from plrf.gaussian import (closed_form, hermite_coefficients, mean_square, mehler, second_moment_and_mean,
                           truncation_degree)
from plrf.inputs import GAUSSIAN, unit_trace_power_law
from plrf.montecarlo import Accumulator

SCALES = {"unit trace": lambda v: unit_trace_power_law(v, 1.31), "raw": lambda v: power_law_spectrum(v, 1.31)}


def rel(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(B)


def setup(scale, v=120, d=40, seed=0):
    H = SCALES[scale](v)
    W = np.random.default_rng(seed).standard_normal((v, d))
    return H, W, sketch_gram(W, H)


def test_mehler_vs_closed_forms():
    for scale in SCALES:
        H, W, G = setup(scale)
        s = np.diag(G)
        L = truncation_degree(G)
        for act in [RELU, HEAVISIDE, BUMP]:
            K, mu = closed_form(G, act)
            A = hermite_coefficients(act, s, L)
            M = mehler(G, A, range(L + 1), diag=np.diag(K))
            err = rel(M, K)
            print(f"  {scale:10s} {act.name:14s} Mehler (L={L}) vs closed form: {err:.1e}   mean: {np.abs(A[:, 0] - mu).max():.1e}")
            assert err < 1e-8 and np.allclose(A[:, 0], mu, atol=1e-10), act.name
            if act is BUMP:  # the quadrature diagonal is right too
                assert np.allclose(mean_square(act, s), np.diag(K), rtol=1e-10)
        for p in (2, 3):
            K, _ = second_moment_and_mean(G, monomial(p))
            err = rel(K, second_moment(G, monomial(p)))
            print(f"  {scale:10s} y^{p}            Mehler vs plrf.exact: {err:.1e}")
            assert err < 1e-10


def test_against_monte_carlo(m=400_000, tol=0.02):
    for scale in SCALES:
        H, W, G = setup(scale, v=200, d=40, seed=1)
        acts = [RELU, HEAVISIDE, BUMP, TANH]
        rng = np.random.default_rng(5)
        acc = Accumulator(W, acts)
        for _ in range(m // 20000):
            acc.add(GAUSSIAN.sample(rng, (20000, len(H))) * np.sqrt(H))
        for act, (Kmc, _) in zip(acts, acc.result()):
            K, _ = second_moment_and_mean(G, act)
            err = rel(Kmc, K)
            print(f"  {scale:10s} {act.name:14s} exact vs Monte Carlo: {err:.4f}")
            assert err < tol, (scale, act.name)


def test_quadrature_converged():
    for scale in SCALES:
        H, W, G = setup(scale)
        s = np.diag(G)
        a1, a2 = hermite_coefficients(TANH, s, 300, 1000), hermite_coefficients(TANH, s, 300, 2000)
        err = np.abs(a1 - a2).max()
        print(f"  {scale:10s} tanh coefficients, 1000 vs 2000 nodes: max difference {err:.1e} (s up to {s.max():.1f})")
        assert err < 1e-10


def test_torch():
    try:
        import torch
    except ImportError:
        print("  torch not installed; skipped")
        return
    H, W, G = setup("raw")
    Gt = torch.as_tensor(G, dtype=torch.float64)
    for act in [RELU, HEAVISIDE, BUMP, TANH]:
        K, mu = second_moment_and_mean(G, act)
        Kt, mut = second_moment_and_mean(Gt, act)
        assert np.allclose(K, Kt.numpy(), rtol=1e-12, atol=1e-14) and np.allclose(mu, np.asarray(mut)), act.name
    print("  numpy and torch agree")


if __name__ == "__main__":
    test_mehler_vs_closed_forms()
    test_quadrature_converged()
    test_against_monte_carlo()
    test_torch()
    print("ok")
