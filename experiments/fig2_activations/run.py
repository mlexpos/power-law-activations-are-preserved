"""Figure 2 (right), redone: random features with non-polynomial activations (ReLU, Heaviside, tanh,
y^2 exp(-y^2)) and Gaussian data.

For each d and each of n_draws sketches W (v = 2d), the exact spectrum of
    K_W = (1/d) E_x[f(W^T x) f(W^T x)^T],   x ~ N(0, sigma^2 H),  H = diag(j^{-alpha}) / tr,
computed without Monte Carlo (plrf.gaussian: closed forms for ReLU, Heaviside and y^2 exp(-y^2), the
Mehler series for tanh).  sigma^2 = E y_i^2 is the pre-activation scale; it matters for tanh and
y^2 exp(-y^2), which are not homogeneous.  Two scales: sigma^2 = 1, and the scale of the submitted
figure (H = diag(j^{-alpha}) unnormalized with v = 10^4, so sigma^2 = sum_{j <= 10^4} j^{-alpha}).

At d = d_decomp the matrix is also split by Hermite degree (plrf.gaussian.mehler):
    K_W = sum_k K^(k),   K^(k)_ij = (1/d) a_k(s_i) a_k(s_j) rho_ij^k,
and the spectra of the partial sums K^(<=L) and of the remainder K^(>8) are stored.

Usage (from the repository root):
    python -m experiments.fig2_activations.run --config quick
    python -m experiments.fig2_activations.run --config paper      # GPU recommended
Output: results/fig2_activations_<config>.npz
"""

import argparse
import json
import os
import time

import numpy as np

from plrf.activations import BUMP, HEAVISIDE, RELU, TANH
from plrf.gaussian import hermite_coefficients, mehler, second_moment_and_mean, truncation_degree
from plrf.inputs import unit_trace_power_law

CONFIGS = {
    "quick": dict(alpha=1.31, ds=[300, 1000], v_over_d=2, n_draws=3, d_decomp=1000),
    "paper": dict(alpha=1.31, ds=[1000, 3000, 10000], v_over_d=2, n_draws=20, d_decomp=3000),
}
ACTS = {"relu": RELU, "heaviside": HEAVISIDE, "tanh": TANH, "bump": BUMP}
PARTIAL = [1, 2, 4, 8]      # Hermite partial sums K^(<=L) stored at d_decomp
REMAINDER = 8               # and the remainder K^(>8)
TAG = 2                     # Figure 2


def scales(alpha):
    return {"unit": 1.0, "old": float((np.arange(1, 10_001, dtype=float) ** (-alpha)).sum())}


def w_seed(d, draw):
    """W ~ numpy.random.default_rng([2, d, draw]).standard_normal((v, d)); the same W for both scales."""
    return [TAG, d, draw]


class Backend:
    def __init__(self, device):
        self.device = device

    def tensor(self, x):
        if self.device == "numpy":
            return x
        import torch
        return torch.as_tensor(x, dtype=torch.float64, device=self.device)

    def eig(self, K):
        if self.device == "numpy":
            return np.linalg.eigvalsh(K)[::-1].copy()
        import torch
        return torch.linalg.eigvalsh(K).flip(0).cpu().numpy()

    def outer(self, a, b):
        if self.device == "numpy":
            return np.outer(a, b)
        import torch
        return torch.outer(a, b)

    def free(self):
        if self.device not in ("numpy", "cpu"):
            import torch
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick", choices=CONFIGS)
    ap.add_argument("--device", default="auto", help="auto | numpy | cpu | cuda")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "numpy"
        except ImportError:
            device = "numpy"
    be = Backend(device)
    sc = scales(cfg["alpha"])
    print(f"config={args.config} {cfg} scales={sc} backend={device}", flush=True)

    lam = {}

    def put(key, x):
        lam.setdefault(key, []).append(x)

    for d in cfg["ds"]:
        v = cfg["v_over_d"] * d
        H = unit_trace_power_law(v, cfg["alpha"])
        for draw in range(cfg["n_draws"]):
            t0 = time.time()
            W = np.random.default_rng(w_seed(d, draw)).standard_normal((v, d))
            G1 = W.T @ (H[:, None] * W)
            del W
            for sname, sig2 in sc.items():
                G = be.tensor(sig2 * G1)
                for a, f in ACTS.items():
                    K, mu = second_moment_and_mean(G, f)
                    put(f"{sname}__{a}__d{d}__K", be.eig(K))
                    put(f"{sname}__{a}__d{d}__C", be.eig(K - be.outer(mu, mu) / d))
                    del K
                    if d == cfg["d_decomp"]:
                        s = np.diag(sig2 * G1)
                        L = truncation_degree(G)
                        A = hermite_coefficients(f, s, L)
                        for Lp in PARTIAL:
                            put(f"{sname}__{a}__d{d}__le{Lp}", be.eig(mehler(G, A, range(Lp + 1)) / d))
                        put(f"{sname}__{a}__d{d}__gt{REMAINDER}",
                            be.eig(mehler(G, A, range(REMAINDER + 1, L + 1)) / d))
                        # energy of each degree, E a_k(s)^2, averaged over neurons
                        put(f"{sname}__{a}__d{d}__energy", (A**2).mean(0)[:64])
                del G
                be.free()
            print(f"d={d}: draw {draw + 1}/{cfg['n_draws']} in {time.time() - t0:.1f}s", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"fig2_activations_{args.config}.npz")
    meta = dict(cfg, config=args.config, backend=device, scales=sc, partial=PARTIAL, remainder=REMAINDER,
                activations={k: f.name for k, f in ACTS.items()},
                key_format="<scale>__<activation>__d<d>__<K|C|le<L>|gt<L>|energy>; K = second moment, "
                           "C = covariance, le<L>/gt<L> = Hermite partial sum / remainder (second moment); "
                           "arrays are (n_draws, d), eigenvalues in decreasing order",
                seed_rule="W ~ numpy.random.default_rng([2, d, draw]).standard_normal((v, d)), shared by both scales")
    np.savez_compressed(out, meta=json.dumps(meta), **{k: np.array(x) for k, x in lam.items()})
    print(f"saved {out}")


if __name__ == "__main__":
    main()
