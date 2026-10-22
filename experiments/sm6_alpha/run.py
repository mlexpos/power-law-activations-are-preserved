"""Supplement SM6: random features when the data exponent is alpha < 1.

For alpha < 1, tr H = sum_j j^{-alpha} grows with v, which the main theorem's proof uses not to happen.
The setup follows the figure in the first version of the supplement: f(y) = y^p, p = 1..4, Gaussian data
with H = diag(j^{-alpha}), and v = d^{1/p + 0.3} (the theorem's dimension condition with epsilon = 0.3).

Two parts:
  reproduce  the original figure's computation: alpha = 0.7, d = 3000, one sketch, the sample covariance
             of f(W^T x) over m = 3000 Monte Carlo samples (not divided by d, as in the original).
  exact      alpha in {0.5, 0.7}, d in {1000, 3000, 10000}, n_draws sketches each: the exact K_W
             (plrf.exact, no Monte Carlo), the population spectrum for p <= 3 (plrf.population, computed
             with no truncation since v is small), and the principal spectrum (p!)^2 H^{1_p} (the sorted
             products H_{s_1} ... H_{s_p}, s_1 < ... < s_p) for every p.

H is not normalized: the activations are monomials, so a scale only multiplies each spectrum by a
constant, and the plots divide by a deterministic constant anyway.

Usage (from the repository root):
    python -m experiments.sm6_alpha.run --config quick
    python -m experiments.sm6_alpha.run --config paper      # GPU recommended
Output: results/sm6_alpha_<config>.npz
"""

import argparse
import json
import os
import time

import numpy as np

from experiments.fig2_theorem.run import spectra_numpy, spectra_torch
from plrf.activations import monomial
from plrf.exact import power_law_spectrum
from plrf.population import population_spectrum_poly, principal_top

CONFIGS = {
    "quick": dict(alphas=[0.5, 0.7], ds=[300, 1000], n_draws=3, eps=0.3, ps=[1, 2, 3, 4],
                  reproduce=dict(alpha=0.7, d=1000, m=1000)),
    "paper": dict(alphas=[0.5, 0.7], ds=[1000, 3000, 10000], n_draws=20, eps=0.3, ps=[1, 2, 3, 4],
                  reproduce=dict(alpha=0.7, d=3000, m=3000)),
}
TAG = 6  # SM6


def v_for(d, p, eps):
    return int(round(d ** (1.0 / p + eps)))


def reproduce(cfg, store):
    """The original figure's computation (Monte Carlo sample covariance), one sketch per p."""
    r = cfg["reproduce"]
    d, m, alpha = r["d"], r["m"], r["alpha"]
    for p in cfg["ps"]:
        v = v_for(d, p, cfg["eps"])
        H = power_law_spectrum(v, alpha)
        rng = np.random.default_rng([TAG, 0, p, d])
        W = rng.standard_normal((v, d))
        Y = ((rng.standard_normal((m, v)) * np.sqrt(H)) @ W) ** p
        Y -= Y.mean(0)
        store[f"repro_p{p}"] = np.linalg.eigvalsh(Y.T @ Y / m)[::-1].copy()
        store[f"repro_principal_p{p}"] = d * principal_top(p, H, d)
        print(f"reproduce p={p}: v={v} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick", choices=CONFIGS)
    ap.add_argument("--part", default="all", choices=["reproduce", "exact", "all"])
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
    print(f"config={args.config} {cfg} backend={device}", flush=True)

    store = {}
    if args.part in ("reproduce", "all"):
        reproduce(cfg, store)
    if args.part in ("exact", "all"):
        for alpha in cfg["alphas"]:
            for d in cfg["ds"]:
                for p in cfg["ps"]:
                    v = v_for(d, p, cfg["eps"])
                    H = power_law_spectrum(v, alpha)
                    act = monomial(p)
                    t0 = time.time()
                    key = f"a{alpha}_p{p}_d{d}"
                    store[f"principal_{key}"] = (factorial_sq(p)) * principal_top(p, H, d)
                    if p <= 3:
                        store[f"pop_{key}"] = population_spectrum_poly(act, H, n=d, head=v)
                    lams = []
                    for draw in range(cfg["n_draws"]):
                        W = np.random.default_rng([TAG, 1, p, d, draw]).standard_normal((v, d))
                        res = spectra_numpy(W, H, [act]) if device == "numpy" else spectra_torch(W, H, [act], device)
                        lams.append(res[0])
                        del W
                    store[f"lam_{key}"] = np.array(lams)
                    print(f"alpha={alpha} d={d} p={p} v={v}: {cfg['n_draws']} sketches in {time.time() - t0:.0f}s", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    suffix = "" if args.part == "all" else f"_{args.part}"
    out = os.path.join(args.outdir, f"sm6_alpha_{args.config}{suffix}.npz")
    meta = dict(cfg, config=args.config, backend=device,
                key_format="lam_a<alpha>_p<p>_d<d>: (n_draws, d) eigenvalues of K_W; pop_...: population spectrum "
                           "(p <= 3); principal_...: (p!)^2 H^{1_p}; repro_p<p>: the original figure's Monte Carlo "
                           "spectrum (not divided by d); repro_principal_p<p>: d H^{1_p}",
                seed_rule="exact: W ~ default_rng([6, 1, p, d, draw]); reproduce: default_rng([6, 0, p, d]) draws W, then x")
    np.savez_compressed(out, meta=json.dumps(meta), **store)
    print(f"saved {out}")


def factorial_sq(p):
    from math import factorial
    return float(factorial(p)) ** 2


if __name__ == "__main__":
    main()
