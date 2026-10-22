"""Figure 2 (left), corrected: the random-feature spectrum tracks the deterministic population
spectrum, as Theorem 2.3 asserts.

For each d and each of n_draws independent sketches W (v = 2d), compute the exact spectrum of
    K_W = (1/d) E_x[f(W^T x) f(W^T x)^T],   x ~ N(0, H),  H = diag(j^{-alpha}) / tr,
for the monomials y, y^2 (also centered), y^3 and the non-monomial polynomials y + y^2/2 + y^3/6 and
He_3(y) = y^3 - 3y (closed form, no Monte Carlo; plrf.exact), and the population spectrum, the
d -> infinity limit for the same v (plrf.population.population_spectrum_poly).

Usage (from the repository root):
    python -m experiments.fig2_theorem.run --config quick      # laptop, a few minutes
    python -m experiments.fig2_theorem.run --config paper      # GPU recommended (d up to 10^4)
Output: results/fig2_theorem_<config>.npz
"""

import argparse
import json
import os
import time

import numpy as np

from plrf.activations import Activation, monomial
from plrf.inputs import unit_trace_power_law
from plrf.population import population_spectrum_poly

CONFIGS = {
    "quick": dict(alpha=1.31, ds=[500, 1000, 2000], v_over_d=2, n_draws=5),
    "paper": dict(alpha=1.31, ds=[1000, 3000, 10000], v_over_d=2, n_draws=20),
}
CUBIC = Activation("y + y^2/2 + y^3/6", (0.0, 1.0, 0.5, 1 / 6))
HE3 = Activation("He_3(y) = y^3 - 3y", (0.0, -3.0, 0.0, 1.0))
ACTIVATIONS = [monomial(1), monomial(2), monomial(2, centered=True), monomial(3), CUBIC, HE3]
KEY = {"y^1": "y1", "y^2": "y2", "y^2 (centered)": "y2c", "y^3": "y3", CUBIC.name: "cub", HE3.name: "he3"}


def seed_for(d, draw):
    """Seed of the sketch W for a given (d, draw); W is always drawn with numpy, so results
    do not depend on the backend used for the linear algebra."""
    return 1_000_003 * d + draw


def spectra_numpy(W, H, acts):
    from plrf.exact import second_moment, sketch_gram
    G = sketch_gram(W, H)
    return [np.linalg.eigvalsh(second_moment(G, a))[::-1] for a in acts]


def spectra_torch(W, H, acts, device):
    """Same computation as spectra_numpy, in float64 on a torch device."""
    import torch
    from math import factorial
    Wt = torch.as_tensor(W, dtype=torch.float64, device=device)
    Ht = torch.as_tensor(H, dtype=torch.float64, device=device)
    G = Wt.T @ (Ht[:, None] * Wt)
    del Wt
    s = torch.diagonal(G).clone()
    d = G.shape[0]
    out = []
    for a in acts:
        K = torch.zeros_like(G)
        for k, B in a.wick_polys().items():
            b = sum(float(B[m]) * s**m for m in range(len(B)))
            K += factorial(k) * torch.outer(b, b) * (G**k if k else 1.0)
        K /= d
        out.append(torch.linalg.eigvalsh(K).flip(0).cpu().numpy())
        del K
    del G
    if device != "cpu":
        torch.cuda.empty_cache()
    return out


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
    print(f"config={args.config} {cfg} backend={device}", flush=True)

    store = {}
    for d in cfg["ds"]:
        v = cfg["v_over_d"] * d
        H = unit_trace_power_law(v, cfg["alpha"])
        t0 = time.time()
        for a in ACTIVATIONS:
            store[f"pop_{KEY[a.name]}_d{d}"] = population_spectrum_poly(a, H, n=d)
        print(f"d={d}: population spectra in {time.time() - t0:.1f}s", flush=True)
        lams = {a.name: [] for a in ACTIVATIONS}
        for draw in range(cfg["n_draws"]):
            t0 = time.time()
            W = np.random.default_rng(seed_for(d, draw)).standard_normal((v, d))
            res = spectra_numpy(W, H, ACTIVATIONS) if device == "numpy" else spectra_torch(W, H, ACTIVATIONS, device)
            for a, lam in zip(ACTIVATIONS, res):
                lams[a.name].append(lam)
            print(f"d={d}: draw {draw + 1}/{cfg['n_draws']} in {time.time() - t0:.1f}s", flush=True)
        for a in ACTIVATIONS:
            store[f"lam_{KEY[a.name]}_d{d}"] = np.array(lams[a.name])

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"fig2_theorem_{args.config}.npz")
    meta = dict(cfg, config=args.config, backend=device, activations=[a.name for a in ACTIVATIONS], H="diag(j^-alpha), unit trace",
                seed_rule="W ~ numpy.random.default_rng(1_000_003*d + draw).standard_normal((v, d))")
    np.savez_compressed(out, meta=json.dumps(meta), **store)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
