"""Polynomial activations and their Wick (Hermite) expansions.

For a centered Gaussian y with variance s, the k-th Wick power is
    :y^k: = s^{k/2} He_k(y / sqrt(s)),
where He_k is the probabilists' Hermite polynomial.  For jointly Gaussian (y, y') with
variances (s, s') and covariance g,
    E[:y^k: :y'^l:] = delta_{kl} k! g^k.                                         (*)
A monomial expands as
    y^r = sum_{m=0}^{floor(r/2)} r! / (m! 2^m (r-2m)!) * s^m * :y^{r-2m}:,
so a polynomial f(y) = sum_r a_r y^r has f(y) = sum_k b_k(s) :y^k:, where each b_k is a
polynomial in s.  Together with (*) this gives closed forms for the second-moment matrix of
random features with Gaussian inputs (see plrf.exact).
"""

from dataclasses import dataclass
from math import factorial

import numpy as np


def monomial_wick_coefficient(r, k):
    """Coefficient c with y^r = ... + c * s^{(r-k)/2} :y^k: + ...  (zero unless r-k is even and >= 0)."""
    if k > r or (r - k) % 2:
        return 0.0
    m = (r - k) // 2
    return factorial(r) / (factorial(m) * 2**m * factorial(k))


@dataclass(frozen=True)
class Activation:
    """A polynomial activation f(y) = sum_r poly[r] y^r.

    If ``centered`` is True, the feature f(y) - E[f(y)] is used instead, i.e. the Wick
    degree-0 component is dropped; the second-moment matrix then becomes the covariance.
    """

    name: str
    poly: tuple
    centered: bool = False

    @property
    def degree(self):
        return max(r for r, a in enumerate(self.poly) if a != 0)

    def wick_polys(self):
        """Return {k: coefficients of b_k as a polynomial in s}, i.e. b_k(s) = sum_m B[m] s^m."""
        out = {}
        for r, a in enumerate(self.poly):
            if a == 0:
                continue
            for k in range(r % 2, r + 1, 2):
                if self.centered and k == 0:
                    continue
                m = (r - k) // 2
                B = out.setdefault(k, np.zeros(self.degree // 2 + 1))
                B[m] += a * monomial_wick_coefficient(r, k)
        return out

    def wick_coefficients(self, s):
        """Evaluate b_k(s) for an array of variances s; returns {k: array like s}."""
        s = np.asarray(s, dtype=float)
        return {k: sum(B[m] * s**m for m in range(len(B))) for k, B in self.wick_polys().items()}

    def homogeneous_wick(self):
        """For f homogeneous of degree p (a monomial, or a single Wick power, possibly centered),
        return (p, {k: beta_k}) with f(y) = sum_k beta_k s^{(p-k)/2} :y^k:.  Used by plrf.population."""
        p = self.degree
        betas = {}
        for k, B in self.wick_polys().items():
            nz = [m for m in range(len(B)) if B[m] != 0]
            if nz != [(p - k) // 2]:
                raise ValueError(f"{self.name} is not homogeneous of degree {p}")
            betas[k] = B[(p - k) // 2]
        return p, betas

    def __call__(self, y):
        """Evaluate the (uncentered) polynomial; centering is applied at the level of matrices."""
        return sum(a * y**r for r, a in enumerate(self.poly))


def monomial(p, centered=False):
    """f(y) = y^p."""
    poly = tuple(1.0 if r == p else 0.0 for r in range(p + 1))
    return Activation(f"y^{p}" + (" (centered)" if centered else ""), poly, centered)


def wick_power(p):
    """The pure degree-p Wick power :y^p: (defined relative to each neuron's own variance).

    Represented as y^p minus its lower Wick components; it is not a fixed polynomial in y,
    so it is only used through its Wick coefficients (beta_p = 1, all others 0)."""
    return _WickPower(f":y^{p}:", tuple(1.0 if r == p else 0.0 for r in range(p + 1)), False)


class _WickPower(Activation):
    def wick_polys(self):
        p = self.degree
        B = np.zeros(p // 2 + 1)
        B[0] = 1.0
        return {p: B}

    def __call__(self, y):
        raise NotImplementedError(":y^p: depends on the variance of y; use the Wick coefficients")


@dataclass(frozen=True)
class Pointwise:
    """A non-polynomial activation, evaluated pointwise on numpy arrays or torch tensors."""

    name: str
    fn: object

    def __call__(self, y):
        return self.fn(y)


def _xp(y):
    if type(y).__module__.startswith("torch"):
        import torch
        return torch
    return np


RELU = Pointwise("ReLU", lambda y: y * (y > 0))
TANH = Pointwise("tanh", lambda y: _xp(y).tanh(y))
HEAVISIDE = Pointwise("Heaviside", lambda y: (y > 0) * 1.0)
BUMP = Pointwise("y^2 exp(-y^2)", lambda y: y * y * _xp(y).exp(-y * y))
