"""Input distributions with independent coordinates, and CIFAR-10.

Synthetic inputs are x = H^{1/2} xi in R^v, where the xi_k are i.i.d., symmetric and standardized
(E xi = 0, E xi^2 = 1), so that Cov(x) = diag(H) for every distribution.  There is no per-sample
normalization.  Since W has i.i.d. N(0, 1) entries, a diagonal H loses no generality: W^T U has
the same law as W^T for any rotation U.

Each distribution provides its standardized cumulants kappa_r (used by plrf.iid for exact
second-moment matrices of polynomial features) and a sampler (used for Monte Carlo).
"""

import os
import pickle
import tarfile
import urllib.request
import warnings
from dataclasses import dataclass
from math import comb, prod

import numpy as np


def cumulants_from_moments(m):
    """Cumulants kappa_1..kappa_n from raw moments m = [m_0 = 1, m_1, ..., m_n]."""
    n = len(m) - 1
    k = [0.0] * (n + 1)
    for r in range(1, n + 1):
        k[r] = m[r] - sum(comb(r - 1, i - 1) * k[i] * m[r - i] for i in range(1, r))
    return k


@dataclass(frozen=True)
class Coordinates:
    """Law of the standardized coordinate xi."""

    name: str
    kind: str          # "gaussian" | "rademacher" | "t"
    nu: float = 0.0    # degrees of freedom, for kind == "t"

    def moment(self, r):
        """E xi^r (infinite if it does not exist)."""
        if r % 2:
            return 0.0
        k = r // 2
        if self.kind == "gaussian":
            return float(prod(range(1, r, 2)))
        if self.kind == "rademacher":
            return 1.0
        # standardized t_nu: xi = t sqrt((nu - 2) / nu), E t^{2k} = nu^k (2k-1)!! / prod_{i<=k} (nu - 2i)
        if self.nu <= r:
            return np.inf
        return (self.nu - 2) ** k * prod(range(1, r, 2)) / prod(self.nu - 2 * i for i in range(1, k + 1))

    def cumulants(self, rmax):
        """{r: kappa_r} for 3 <= r <= rmax, nonzero only (kappa_2 = 1 always)."""
        m = [self.moment(r) for r in range(rmax + 1)]
        if not np.isfinite(m[-1]):
            raise ValueError(f"{self.name}: moment of order {rmax} is infinite")
        k = cumulants_from_moments(m)
        return {r: k[r] for r in range(3, rmax + 1) if abs(k[r]) > 1e-12}

    def sample(self, rng, shape):
        if self.kind == "gaussian":
            return rng.standard_normal(shape)
        if self.kind == "rademacher":
            return 2.0 * rng.integers(0, 2, size=shape) - 1.0
        return rng.standard_t(self.nu, size=shape) * np.sqrt((self.nu - 2) / self.nu)


GAUSSIAN = Coordinates("Gaussian", "gaussian")
RADEMACHER = Coordinates("Rademacher", "rademacher")


def student_t(nu):
    return Coordinates(f"Student-t({nu:g})", "t", float(nu))


def unit_trace_power_law(v, alpha):
    """H_j = j^{-alpha} / sum_k k^{-alpha}, so that E|x|^2 = tr H = 1 and E y_i^2 = E s_i = 1."""
    H = np.arange(1, v + 1, dtype=float) ** (-alpha)
    return H / H.sum()


# ---- CIFAR-10 ----

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def load_cifar10(data_dir="data"):
    """All 60,000 CIFAR-10 images (train and test) as an (N, 3072) float64 array in [0, 1].
    Downloads the python version of the dataset into data_dir if it is not there."""
    root = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.isdir(root):
        os.makedirs(data_dir, exist_ok=True)
        tgz = os.path.join(data_dir, "cifar-10-python.tar.gz")
        print(f"downloading {CIFAR_URL}", flush=True)
        urllib.request.urlretrieve(CIFAR_URL, tgz)
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(data_dir)
    batches = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
    X = []
    for b in batches:
        with open(os.path.join(root, b), "rb") as f, warnings.catch_warnings():
            warnings.simplefilter("ignore")   # numpy >= 2.4 warns about the old pickle format
            X.append(pickle.load(f, encoding="bytes")[b"data"])
    return np.vstack(X).astype(np.float64) / 255.0


def preprocess_cifar(X):
    """Subtract the per-pixel mean and divide by one global constant so that the mean of |x|^2 over
    the dataset is 1 (the same convention as tr H = 1 for the synthetic inputs).  Returns the
    preprocessed images and their covariance Sigma (so Sigma has unit trace)."""
    X = X - X.mean(axis=0)
    X /= np.sqrt((X**2).sum(axis=1).mean())
    Sigma = X.T @ X / X.shape[0]
    return X, Sigma
