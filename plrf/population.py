"""The deterministic population spectrum: the d -> infinity limit of spec(K_W) for fixed v and H.

(K_W)_ij = kappa(w_i, w_j) / d, where w_i ~ N(0, I_v) are the i.i.d. columns of W and
    kappa(w, w') = E_x[f(w^T x) f(w'^T x)],   x ~ N(0, H).
As d -> infinity the eigenvalues of K_W converge to those of the integral operator
    (T g)(w) = E_{w'}[kappa(w, w') g(w')]   on L^2(N(0, I_v)).

For f homogeneous of degree p, f(y) = sum_k beta_k s^{(p-k)/2} :y^k: (plrf.activations), and
    kappa(w, w') = sum_k k! beta_k^2 (s_w s_w')^{(p-k)/2} (w^T H w')^k,   s_w = w^T H w,
a polynomial of degree p in each argument.  Writing kappa(w, w') = phi(w)^T C phi(w') in the
basis phi of degree-p monomials w^n, the nonzero eigenvalues of T are those of C M, where
M = E[phi phi^T] is the Gram matrix of the monomials under N(0, I).  Both C and M are block
diagonal by the set of coordinates with odd exponent (sign-flip symmetry), which gives:

* squarefree monomials (odd set of size p): orthonormal and decoupled, eigenvalues
  (p! beta_p)^2 H_{s_1} ... H_{s_p}, s_1 < ... < s_p  -- the principal term of the paper;
* the remaining blocks (implemented here for p <= 3):
    p = 2, odd set {}:  monomials w_a^2;       C = 2 beta_2^2 diag(H_a^2) + beta_0^2 h h^T
    p = 3, odd set {t}: monomials w_a^2 w_t (a != t), w_t^3;
                        C = diag(18 beta_3^2 H_a^2 H_t, 6 beta_3^2 H_t^3) + beta_1^2 H_t h h^T
  with h the coefficient vector of s_w (resp. s_w w_t) in that basis, and M given by Gaussian
  moments (E w^2 = 1, E w^4 = 3, E w^6 = 15).

Numerics.  In the non-squarefree blocks the diagonal parts decay like H_a^2, so only the first
``head`` indices a are kept explicitly; the rank-one direction h (which carries tr H and decays
only like H_a) is kept exactly, with all sums over a in [v].  The dropped coordinates have
eigenvalue contributions of order H_head^2 (relative to the block scale), far below the part of
the spectrum that is compared with K_W.  See tests/test_population.py.
"""

import heapq
from math import factorial

import numpy as np


def principal_top(p, H, n):
    """Largest n values of H_{s_1} ... H_{s_p} over 1 <= s_1 < ... < s_p <= len(H), in decreasing
    order (best-first search over the lattice; H must be non-increasing)."""
    v = len(H)
    logH = np.log(H)
    start = tuple(range(p))
    heap = [(-logH[list(start)].sum(), start)]
    seen = {start}
    out = []
    while heap and len(out) < n:
        neg, t = heapq.heappop(heap)
        out.append(np.exp(-neg))
        for k in range(p):
            u = list(t)
            u[k] += 1
            if u[k] < v and (k == p - 1 or u[k] < u[k + 1]):
                u = tuple(u)
                if u not in seen:
                    seen.add(u)
                    heapq.heappush(heap, (-logH[list(u)].sum(), u))
    return np.array(out)


def _block_p2(H, beta2, beta0, head):
    """Odd set {}: monomials w_a^2.  M_ab = 1 + 2 delta_ab."""
    S = min(head, len(H))
    Hh = H[:S]
    trH, trH2 = H.sum(), (H**2).sum()
    sqD = np.sqrt(2.0) * abs(beta2) * Hh                       # sqrt(D_a), D_a = 2 beta2^2 H_a^2
    B_DD = np.outer(sqD, sqD) * (1.0 + 2.0 * np.eye(S))
    if beta0 == 0:
        return np.linalg.eigvalsh(B_DD)
    Mh = trH + 2.0 * Hh                                        # (M h)_a, h_a = H_a
    B = np.empty((S + 1, S + 1))
    B[:S, :S] = B_DD
    B[:S, S] = B[S, :S] = sqD * abs(beta0) * Mh
    B[S, S] = beta0**2 * (trH**2 + 2.0 * trH2)                 # beta0^2 h^T M h
    return np.linalg.eigvalsh(B)


def _blocks_p3(H, beta3, beta1, head, chunk=512):
    """Odd sets {t}, t = 1..v: monomials w_a^2 w_t (a != t) and w_t^3.
    M: (a,b) -> 1 + 2 delta_ab, (a, t^3) -> 3, (t^3, t^3) -> 15."""
    v = len(H)
    S = min(head, v)
    trH, trH2 = H.sum(), (H**2).sum()
    b3, b1 = abs(beta3), abs(beta1)
    out = []

    def block_mats(Ht, Ha):
        # Ht: (T,), Ha: (T, n) head values H_a for a != t.  Returns batched symmetric matrices.
        T, n = Ha.shape
        size = n + 1 + (1 if b1 else 0)
        B = np.zeros((T, size, size))
        sqD = np.sqrt(18.0) * b3 * Ha * np.sqrt(Ht)[:, None]                    # sqrt(18 b3^2 H_a^2 H_t)
        B[:, :n, :n] = sqD[:, :, None] * sqD[:, None, :] * (1.0 + 2.0 * np.eye(n))[None]
        sqDc = np.sqrt(6.0) * b3 * Ht**1.5                                       # sqrt(6 b3^2 H_t^3)
        B[:, :n, n] = B[:, n, :n] = 3.0 * sqD * sqDc[:, None]
        B[:, n, n] = 15.0 * sqDc**2
        if b1:
            rest = trH - Ht                                                      # sum_{a != t} H_a
            Mh_a = rest[:, None] + 2.0 * Ha + 3.0 * Ht[:, None]
            Mh_c = 3.0 * rest + 15.0 * Ht
            hMh = rest**2 + 2.0 * (trH2 - Ht**2) + 6.0 * Ht * rest + 15.0 * Ht**2
            scale = b1 * np.sqrt(Ht)                                             # sqrt(beta1^2 H_t)
            B[:, :n, n + 1] = B[:, n + 1, :n] = sqD * scale[:, None] * Mh_a
            B[:, n, n + 1] = B[:, n + 1, n] = sqDc * scale * Mh_c
            B[:, n + 1, n + 1] = scale**2 * hMh
        return B

    # t inside the head: exclude a = t
    for t in range(min(S, v)):
        idx = np.r_[0:t, t + 1:S]
        out.append(np.linalg.eigvalsh(block_mats(H[t:t + 1], H[idx][None, :])[0]))
    # t outside the head: all a in the head; batch over t
    Ht_all = H[S:]
    for i in range(0, len(Ht_all), chunk):
        Ht = Ht_all[i:i + chunk]
        Ha = np.broadcast_to(H[:S], (len(Ht), S))
        out.append(np.linalg.eigvalsh(block_mats(Ht, Ha)).ravel())
    return np.concatenate(out)


def population_spectrum(activation, H, n, head=150):
    """Top-n eigenvalues (decreasing) of the d -> infinity population operator for a homogeneous
    activation of degree p <= 3 (monomials y^p, possibly centered, and Wick powers :y^p:)."""
    p, betas = activation.homogeneous_wick()
    if p == 1:
        vals = betas.get(1, 0.0) ** 2 * H
    elif p == 2:
        vals = np.r_[(2.0 * betas.get(2, 0.0)) ** 2 * principal_top(2, H, n),
                     _block_p2(H, betas.get(2, 0.0), betas.get(0, 0.0), head)]
    elif p == 3:
        vals = np.r_[(6.0 * betas.get(3, 0.0)) ** 2 * principal_top(3, H, n),
                     _blocks_p3(H, betas.get(3, 0.0), betas.get(1, 0.0), head)]
    else:
        raise NotImplementedError("population spectrum implemented for degree p <= 3")
    vals = np.sort(vals)[::-1]
    return vals[:n]


# ---- general polynomials of degree <= 3 (not necessarily homogeneous) ----
#
# f(y) = sum_k b_k(s) :y^k: with b_k(s) = sum_m B_k[m] s^m (plrf.activations), so
#     kappa(w, w') = sum_k k! b_k(s_w) b_k(s_w') (w^T H w')^k,
# a polynomial of degree <= p in each argument, no longer homogeneous.  The parity blocks are as above,
# but their bases now mix degrees:
#   odd set {}:      1, w_a^2                    from k = 0 (b_0(s_w) = B_0[0] + B_0[1] s_w) and k = 2
#   odd set {t}:     w_t, w_a^2 w_t, w_t^3        from k = 1 (b_1(s_w) w_t) and k = 3
#   odd set {a, b}:  w_a w_b                      from k = 2:  eigenvalue (2 B_2[0])^2 H_a H_b
#   odd set {a,b,c}: w_a w_b w_c                  from k = 3:  eigenvalue (6 B_3[0])^2 H_a H_b H_c
# In the first two the coefficient matrix C and the Gaussian Gram matrix M are both dense, so each block
# is solved as eig(C M) = eig(L^T C L) with M = L L^T.  As in the homogeneous case, only the first `head`
# indices a are kept explicitly; the remaining ones enter only through the direction
# r = sum_{a > head} H_a w_a^2 (times w_t), which carries their contribution to s_w exactly.


def _wick_B(activation, k, m):
    B = activation.wick_polys().get(k)
    return float(B[m]) if B is not None and m < len(B) else 0.0


def _gen_eig(C, M):
    L = np.linalg.cholesky(M)
    return np.linalg.eigvalsh(np.swapaxes(L, -1, -2) @ C @ L)


def _even_block(H, b00, b01, b2, head):
    v = len(H)
    S = min(head, v)
    Hh = H[:S]
    T, Q = H[S:].sum(), (H[S:] ** 2).sum()
    tail = S < v
    n = 1 + S + (1 if tail else 0)
    g = np.r_[b00, b01 * Hh, [b01] if tail else []]
    C = np.outer(g, g)
    C[1:S + 1, 1:S + 1] += np.diag(2.0 * b2**2 * Hh**2)
    M = np.ones((n, n))
    M[1:S + 1, 1:S + 1] += 2.0 * np.eye(S)
    if tail:
        M[-1, :] = M[:, -1] = T
        M[-1, -1] = T**2 + 2.0 * Q
    return _gen_eig(C, M)


def _odd_blocks(H, b10, b11, b3, head, chunk=256):
    v = len(H)
    S = min(head, v)
    T_S, Q_S = H[S:].sum(), (H[S:] ** 2).sum()
    out = []

    def solve(Ht, Ha, T, Q, tail):
        # batched over the first axis; Ha: (B, n_head) head values with a != t
        Bn, nh = Ha.shape
        n = 2 + nh + (1 if tail else 0)        # order: w_t, w_a^2 w_t (a in head), w_t^3, [r_t]
        g = np.zeros((Bn, n))
        g[:, 0] = b10
        g[:, 1:1 + nh] = b11 * Ha
        g[:, 1 + nh] = b11 * Ht
        if tail:
            g[:, -1] = b11
        C = Ht[:, None, None] * g[:, :, None] * g[:, None, :]
        idx = np.arange(1, 1 + nh)
        C[:, idx, idx] += 18.0 * b3**2 * Ha**2 * Ht[:, None]
        C[:, 1 + nh, 1 + nh] += 6.0 * b3**2 * Ht**3
        M = np.ones((Bn, n, n))
        M[:, idx, idx] += 2.0
        M[:, 0, 1 + nh] = M[:, 1 + nh, 0] = 3.0
        M[:, 1:1 + nh, 1 + nh] = M[:, 1 + nh, 1:1 + nh] = 3.0
        M[:, 1 + nh, 1 + nh] = 15.0
        if tail:
            M[:, -1, :] = M[:, :, -1] = T[:, None]
            M[:, -1, 1 + nh] = M[:, 1 + nh, -1] = 3.0 * T
            M[:, -1, -1] = T**2 + 2.0 * Q
        return _gen_eig(C, M).ravel()

    for t in range(S):                          # t inside the head: exclude a = t
        idx = np.r_[0:t, t + 1:S]
        out.append(solve(H[t:t + 1], H[idx][None, :], np.array([T_S]), np.array([Q_S]), S < v))
    for i in range(S, v, chunk):                # t in the tail: batched
        Ht = H[i:i + chunk]
        out.append(solve(Ht, np.broadcast_to(H[:S], (len(Ht), S)), T_S - Ht, Q_S - Ht**2, v - S > 1))
    return np.concatenate(out)


def population_spectrum_poly(activation, H, n, head=150):
    """Top-n eigenvalues (decreasing) of the d -> infinity population operator for any polynomial
    activation of degree p <= 3 (constant, linear and quadratic terms allowed; centered or not).
    For monomials it agrees with population_spectrum (tests/test_population.py)."""
    p = activation.degree
    if p > 3:
        raise NotImplementedError("population spectrum implemented for degree p <= 3")
    b00, b01 = _wick_B(activation, 0, 0), _wick_B(activation, 0, 1)
    b10, b11 = _wick_B(activation, 1, 0), _wick_B(activation, 1, 1)
    b2, b3 = _wick_B(activation, 2, 0), _wick_B(activation, 3, 0)
    if p <= 1:      # kappa = b00^2 + b10^2 w^T H w': eigenvalues b00^2 and b10^2 H_t
        vals = np.sort(np.r_[b00**2, b10**2 * H])[::-1]
        return vals[:n]
    vals = [np.zeros(0)]
    if b00 or b01 or b2:
        vals.append(_even_block(H, b00, b01, b2, head))
    if b10 or b11 or b3:
        vals.append(_odd_blocks(H, b10, b11, b3, head))
    if b2:
        vals.append((2.0 * b2) ** 2 * principal_top(2, H, n))
    if b3:
        vals.append((6.0 * b3) ** 2 * principal_top(3, H, n))
    vals = np.sort(np.concatenate(vals))[::-1]
    return vals[:n]
