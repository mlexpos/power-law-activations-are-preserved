"""Checks of the exact second-moment matrices for non-Gaussian i.i.d. inputs (plrf.iid), of the
exact ReLU kernel, and of the Monte Carlo accumulator.

1. Gaussian coordinates: plrf.iid reproduces plrf.exact.second_moment to rounding error.
2. Standardized cumulants of Rademacher and Student-t against known values.
3. Rademacher and Student-t coordinates: plrf.iid agrees with Monte Carlo over x.  (Student-t
   with nu = 15 for degree 3, since the Monte Carlo variance needs 4p moments.)
4. ReLU: the arc-cosine formula agrees with Monte Carlo.
5. The torch and numpy Monte Carlo backends agree.

Run:  python -m tests.test_iid
"""

import numpy as np

from plrf.activations import RELU, Activation, monomial
from plrf.exact import power_law_spectrum, relu_second_moment, second_moment, sketch_gram
from plrf.iid import second_moment_iid
from plrf.inputs import GAUSSIAN, RADEMACHER, student_t
from plrf.montecarlo import Accumulator

CUBIC = Activation("1 - y + 0.5 y^2 + 0.2 y^3", (1.0, -1.0, 0.5, 0.2))
CUBIC_C = Activation("1 - y + 0.5 y^2 + 0.2 y^3 (centered)", (1.0, -1.0, 0.5, 0.2), True)


def mc(W, H, coords, acts, m, seed, batch=20000):
    rng = np.random.default_rng(seed)
    acc = Accumulator(W, acts)
    for _ in range(m // batch):
        acc.add(coords.sample(rng, (batch, len(H))) * np.sqrt(H))
    return acc.result()


def rel(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(B)


def test_gaussian_reduces_to_exact():
    rng = np.random.default_rng(0)
    v, d = 50, 20
    H = power_law_spectrum(v, 1.31)
    W = rng.standard_normal((v, d))
    G = sketch_gram(W, H)
    for act in [monomial(1), monomial(2), monomial(3), monomial(4), monomial(2, centered=True), CUBIC, CUBIC_C]:
        err = rel(second_moment_iid(W, H, act, GAUSSIAN), second_moment(G, act))
        print(f"  Gaussian, {act.name:36s} iid vs exact: {err:.1e}")
        assert err < 1e-12, act.name


def test_cumulants():
    assert np.isclose(RADEMACHER.cumulants(6)[4], -2) and np.isclose(RADEMACHER.cumulants(6)[6], 16)
    k = student_t(9).cumulants(6)
    assert np.isclose(k[4], 6 / 5) and np.isclose(k[6], 16)        # kappa_4 = 6 / (nu - 4)
    assert GAUSSIAN.cumulants(8) == {}
    try:
        student_t(5).cumulants(6)
        raise AssertionError("t(5) has no sixth moment")
    except ValueError:
        pass
    # the samplers are standardized
    rng = np.random.default_rng(3)
    for c in [GAUSSIAN, RADEMACHER, student_t(9)]:
        x = c.sample(rng, 2_000_000)
        assert abs(x.var() - 1) < 0.01, c.name


def test_against_monte_carlo(tol=0.03):
    rng = np.random.default_rng(1)
    v, d, m = 200, 40, 400_000
    H = power_law_spectrum(v, 1.31)
    H /= H.sum()
    W = rng.standard_normal((v, d))
    cases = [(RADEMACHER, [monomial(2), monomial(3), CUBIC, CUBIC_C]),
             (student_t(9), [monomial(2), monomial(2, centered=True)]),
             (student_t(15), [monomial(3), CUBIC])]
    for seed, (coords, acts) in enumerate(cases):
        res = mc(W, H, coords, acts, m, seed=100 + seed)
        for act, (K, Kc) in zip(acts, res):
            err = rel(K if not act.centered else Kc, second_moment_iid(W, H, act, coords))
            # the non-Gaussian correction itself, to show the test can tell the difference
            gap = rel(second_moment(sketch_gram(W, H), act), second_moment_iid(W, H, act, coords))
            print(f"  {coords.name:14s} {act.name:36s} MC error {err:.4f}   (Gaussian formula off by {gap:.3f})")
            assert err < tol, (coords.name, act.name)


def test_relu(tol=0.02):
    rng = np.random.default_rng(2)
    v, d, m = 200, 40, 400_000
    H = power_law_spectrum(v, 1.31)
    W = rng.standard_normal((v, d))
    G = sketch_gram(W, H)
    (K, Kc), = mc(W, H, GAUSSIAN, [RELU], m, seed=7)
    for name, est, ex in [("ReLU", K, relu_second_moment(G)), ("ReLU (centered)", Kc, relu_second_moment(G, True))]:
        err = rel(est, ex)
        print(f"  Gaussian, {name:36s} arc-cosine vs MC: {err:.4f}")
        assert err < tol


def test_torch_backend():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  torch not installed; skipped")
        return
    rng = np.random.default_rng(4)
    W = rng.standard_normal((30, 10))
    X = rng.standard_normal((500, 30))
    a, b = Accumulator(W, [RELU, monomial(2)]), Accumulator(W, [RELU, monomial(2)], device="cpu")
    a.add(X); b.add(X)
    for (K1, C1), (K2, C2) in zip(a.result(), b.result()):
        assert np.allclose(K1, K2) and np.allclose(C1, C2)
    print("  numpy and torch accumulators agree")


if __name__ == "__main__":
    test_gaussian_reduces_to_exact()
    test_cumulants()
    test_against_monte_carlo()
    test_relu()
    test_torch_backend()
    print("ok")
