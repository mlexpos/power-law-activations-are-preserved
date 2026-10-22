"""Checks of the population spectrum (the d -> infinity limit of spec K_W).

1. Brute force, small v: expand kappa(w, w') = sum_k k! beta_k^2 (s_w s_w')^m (w^T H w')^k
   (m = (p-k)/2) in *all* degree-p monomials, build the coefficient matrix C and the exact
   Gaussian Gram matrix M of the monomials, and compare eig(C M) with plrf.population, which
   uses the block decomposition instead.  This checks the block formulas exactly.
2. Small v: the nonzero eigenvalues of K_W at large d (plrf.exact, itself checked against
   Monte Carlo in test_exact.py) are close to the population spectrum.  Kernel matrices with
   polynomial kernels converge slowly (kappa(w, w) is heavy-tailed), so this is a loose check.
3. The head truncation in the non-squarefree blocks does not move the top of the spectrum.

Run:  python -m tests.test_population
"""

from itertools import combinations_with_replacement
from math import factorial

import numpy as np

from plrf.activations import Activation, monomial, wick_power
from plrf.exact import power_law_spectrum, second_moment, sketch_gram
from plrf.population import population_spectrum, population_spectrum_poly

ACTS = [monomial(1), monomial(2), monomial(2, centered=True), monomial(3), wick_power(2), wick_power(3)]


def _multi_indices(v, deg):
    out = []
    for c in combinations_with_replacement(range(v), deg):
        n = np.zeros(v, dtype=int)
        for a in c:
            n[a] += 1
        out.append(tuple(n))
    return out


def _multinomial(n):
    r = factorial(sum(n))
    for x in n:
        r //= factorial(x)
    return r


def brute_force_population(act, H):
    p, betas = act.homogeneous_wick()
    v = len(H)
    basis = _multi_indices(v, p)
    pos = {n: i for i, n in enumerate(basis)}
    C = np.zeros((len(basis), len(basis)))
    for k, beta in betas.items():
        m = (p - k) // 2
        for n in _multi_indices(v, k):
            cn = _multinomial(n) * np.prod(H ** np.array(n))
            for q in _multi_indices(v, m):
                cq = _multinomial(q) * np.prod(H ** np.array(q))
                for q2 in _multi_indices(v, m):
                    cq2 = _multinomial(q2) * np.prod(H ** np.array(q2))
                    A = tuple(np.array(n) + 2 * np.array(q))
                    B = tuple(np.array(n) + 2 * np.array(q2))
                    C[pos[A], pos[B]] += factorial(k) * beta**2 * cn * cq * cq2
    dfact = lambda e: 0 if e % 2 else np.prod(np.arange(e - 1, 0, -2)) if e > 0 else 1
    M = np.array([[np.prod([dfact(a + b) for a, b in zip(A, B)]) for B in basis] for A in basis], float)
    ev = np.linalg.eigvals(C @ M).real
    return np.sort(ev[ev > 1e-12 * ev.max()])[::-1]


def brute_force_population_poly(act, H):
    """All nonzero eigenvalues of C M for a general polynomial activation, in the basis of all monomials
    of degree <= p: kappa(w, w') = sum_k k! b_k(s_w) b_k(s_w') (w^T H w')^k, b_k(s) = sum_m B_k[m] s^m."""
    p, v = act.degree, len(H)
    basis = [n for deg in range(p + 1) for n in _multi_indices(v, deg)]
    pos = {n: i for i, n in enumerate(basis)}
    C = np.zeros((len(basis), len(basis)))
    for k, B in act.wick_polys().items():
        # coefficient vector of b_k(s_w) w^n, for each multi-index n with |n| = k
        for n in _multi_indices(v, k):
            cn = _multinomial(n) * np.prod(H ** np.array(n))
            vec = np.zeros(len(basis))
            for m, Bm in enumerate(B):
                if Bm == 0:
                    continue
                for q in _multi_indices(v, m):
                    vec[pos[tuple(np.array(n) + 2 * np.array(q))]] += Bm * _multinomial(q) * np.prod(H ** np.array(q))
            C += factorial(k) * cn * np.outer(vec, vec)
    dfact = lambda e: 0 if e % 2 else np.prod(np.arange(e - 1, 0, -2)) if e > 0 else 1
    M = np.array([[np.prod([dfact(a + b) for a, b in zip(A, B)]) for B in basis] for A in basis], float)
    ev = np.linalg.eigvals(C @ M).real
    return np.sort(ev[ev > 1e-12 * ev.max()])[::-1]


POLYS = [Activation("1 - y + 0.5 y^2 + 0.2 y^3", (1.0, -1.0, 0.5, 0.2)),
         Activation("y + y^2/2 + y^3/6", (0.0, 1.0, 0.5, 1 / 6)),
         Activation("He_3 = y^3 - 3y", (0.0, -3.0, 0.0, 1.0)),
         Activation("He_3 (centered)", (0.0, -3.0, 0.0, 1.0), True),
         Activation("y^2 + y (centered)", (0.0, 1.0, 1.0), True),
         monomial(2), monomial(3)]


def test_brute_force_poly(tol=1e-8):
    for H in [power_law_spectrum(5, 1.31), power_law_spectrum(5, 0.7)]:
        for act in POLYS:
            bf = brute_force_population_poly(act, H)
            for head in (5, 3):         # head = v is exact; head < v exercises the tail direction
                pop = population_spectrum_poly(act, H, n=len(bf), head=head)
                if head == 5:
                    rel = np.abs(pop[: len(bf)] / bf - 1).max()
                    print(f"  {act.name:28s} general block method vs brute force ({len(bf)} eigenvalues): {rel:.1e}")
                    assert rel < tol, act.name
    # the general method agrees with the homogeneous one on monomials, at full size
    H = power_law_spectrum(2000, 1.31)
    for act in [monomial(1), monomial(2), monomial(3), monomial(2, centered=True)]:
        a, b = population_spectrum_poly(act, H, 1000), population_spectrum(act, H, 1000)
        rel = np.abs(a / b - 1).max()
        print(f"  {act.name:28s} general vs homogeneous block method (v = 2000, top 1000): {rel:.1e}")
        assert rel < 1e-8, act.name


def test_brute_force(tol=1e-8):
    H = power_law_spectrum(6, 1.31)
    for act in ACTS:
        bf = brute_force_population(act, H)
        pop = population_spectrum(act, H, n=len(bf), head=6)
        rel = np.abs(pop[: len(bf)] / bf - 1).max()
        print(f"  {act.name:16s} block method vs brute force (all {len(bf)} eigenvalues): max rel diff {rel:.1e}")
        assert rel < tol, act.name


def test_small_v_against_large_d(tol=0.15):
    rng = np.random.default_rng(2)
    v, d = 6, 6000
    H = power_law_spectrum(v, 1.31)
    W = rng.standard_normal((v, d))
    G = sketch_gram(W, H)
    for act in ACTS:
        pop = population_spectrum(act, H, n=200, head=v)
        pop = pop[pop > 1e-12 * pop[0]]
        lam = np.linalg.eigvalsh(second_moment(G, act))[::-1][: len(pop)]
        k = min(len(pop), 15)
        rel = np.abs(lam[:k] / pop[:k] - 1).max()
        print(f"  {act.name:16s} rank {len(pop):3d}; max |lambda_j(K_W)/pop_j - 1| over top {k}: {rel:.3f}")
        assert rel < tol, act.name


def test_head_truncation():
    H = power_law_spectrum(4000, 1.31)
    for act in [monomial(2), monomial(3), wick_power(3)]:
        a = population_spectrum(act, H, n=2000, head=150)   # the default used by the experiments
        b = population_spectrum(act, H, n=2000, head=400)
        rel = np.abs(a / b - 1).max()
        print(f"  {act.name:16s} head 150 vs 400, max relative change in top 2000: {rel:.2e}")
        assert rel < 1e-3, act.name


def test_head_truncation_poly():
    H = power_law_spectrum(4000, 1.31)
    H /= H.sum()
    for act in POLYS[1:3]:
        a = population_spectrum_poly(act, H, n=2000, head=150)
        b = population_spectrum_poly(act, H, n=2000, head=400)
        rel = np.abs(a / b - 1).max()
        print(f"  {act.name:28s} head 150 vs 400 (tr H = 1), max relative change in top 2000: {rel:.2e}")
        assert rel < 1e-3, act.name


if __name__ == "__main__":
    test_brute_force_poly()
    test_brute_force()
    test_small_v_against_large_d()
    test_head_truncation()
    test_head_truncation_poly()
    print("ok")
