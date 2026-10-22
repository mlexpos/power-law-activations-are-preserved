"""The floor of y^2 exp(-y^2): compute and plot.

Write K_W = K^(<=2) + K^(>=3), the Hermite components of degree at most 2 and at least 3
(plrf.gaussian: the full matrix in closed form, the low part from the Mehler series, the high part as the
difference).  For large degree k the correlations rho_ij^k off the diagonal are negligible, so K^(>=3) is
close to (1/d) diag(sum_{k>=3} a_k(s_i)^2): white noise of height about E_{>=3} / d, where
E_{>=3} = mean_i (E f(y_i)^2 - a_0(s_i)^2 - a_1(s_i)^2 - a_2(s_i)^2) is computed from the Hermite
coefficients, with no fitting.

For each scale and each d of the main experiment: E_{>=3} for every sketch (same seeds as run.py).  At
d = d_decomp: the spectra of K_W, K^(<=2) and K^(>=3) for every sketch.

Figure fig2_activations_floor_<config>: rows = scales.  Left: lambda_j(K_W) for each d (from the main
results) with the predicted floor E_{>=3}/d (dashed, same color).  Right: at d = d_decomp, the spectra of
K_W (black), K^(<=2) (blue) and K^(>=3) (orange), and the predicted floor (grey).

Usage:  python -m experiments.fig2_activations.floor --config paper     (needs results from run.py)
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.fig2_activations.run import CONFIGS, scales, w_seed
from plrf.activations import BUMP
from plrf.gaussian import closed_form, hermite_coefficients, mehler
from plrf.inputs import unit_trace_power_law

D_COLORS = ["#86b6ef", "#2a78d6", "#104281"]


def compute(cfg, R):
    out = {}
    for sname, sig2 in scales(cfg["alpha"]).items():
        for d in cfg["ds"]:
            v = cfg["v_over_d"] * d
            H = unit_trace_power_law(v, cfg["alpha"])
            E3 = []
            for draw in range(cfg["n_draws"]):
                W = np.random.default_rng(w_seed(d, draw)).standard_normal((v, d))
                s = sig2 * (H @ W**2)
                A = hermite_coefficients(BUMP, s, 2)
                Ef2 = s**2 * 3 / (1 + 4 * s) ** 2.5          # E[y^4 e^{-2y^2}], y ~ N(0, s)
                E3.append(np.mean(Ef2 - (A**2).sum(1)))
                if d == cfg["d_decomp"]:
                    G = sig2 * (W.T @ (H[:, None] * W))
                    K, _ = closed_form(G, BUMP)
                    Klow = mehler(G, A, range(3))
                    for key, M in [("full", K), ("low", Klow), ("high", K - Klow)]:
                        out.setdefault(f"{sname}__{key}", []).append(np.linalg.eigvalsh(M / d)[::-1])
            out[f"{sname}__E3__d{d}"] = np.array(E3)
            print(f"scale {sname}, d={d}: E_>=3 = {np.median(E3):.4f}", flush=True)
    return {k: np.array(v) for k, v in out.items()}


def band(ax, x, arr, color, q=(10, 50, 90), **kw):
    lo, med, hi = np.percentile(arr, q, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, lw=0)
    ax.plot(x, med, color=color, **kw)


def plot(cfg, R, F, outdir, config):
    sc = scales(cfg["alpha"])
    fig, axes = plt.subplots(len(sc), 2, figsize=(11, 4.3 * len(sc)), gridspec_kw=dict(hspace=0.35, wspace=0.22))
    dd = cfg["d_decomp"]
    for r, (sname, sig2) in enumerate(sc.items()):
        ax, axh = axes[r]
        for d, col in zip(cfg["ds"], D_COLORS[-len(cfg["ds"]):]):
            band(ax, np.arange(1, d + 1), R[f"{sname}__bump__d{d}__K"], col, lw=1.5, label=f"$d={d}$")
            ax.axhline(np.median(F[f"{sname}__E3__d{d}"]) / d, color=col, ls="--", lw=1)
        ax.plot([], [], "k--", lw=1, label=r"predicted floor $E_{\geq 3}/d$")
        j = np.arange(1, dd + 1)
        band(axh, j, F[f"{sname}__full"], "#1f1f1e", lw=1.8, label="$K_W$")
        band(axh, j, F[f"{sname}__low"], "#2a78d6", lw=1.4, label=r"Hermite degrees $\leq 2$")
        band(axh, j, F[f"{sname}__high"], "#eb6834", lw=1.4, ls=(0, (4, 1.5)), label=r"Hermite degrees $\geq 3$")
        axh.axhline(np.median(F[f"{sname}__E3__d{dd}"]) / dd, color="0.5", ls="--", lw=1,
                    label=r"predicted floor $E_{\geq 3}/d$")
        lo = np.percentile(F[f"{sname}__full"][:, -1], 5) / 3
        for a, t in [(ax, f"(a) spectra, $\\sigma^2={sig2:.2f}$"), (axh, f"(b) Hermite split at $d={dd}$, $\\sigma^2={sig2:.2f}$")]:
            a.set_xscale("log"); a.set_yscale("log")
            a.grid(alpha=0.2, which="both"); a.set_xlabel("$j$")
            a.set_title(t, fontsize=10, loc="left")
            a.legend(fontsize=8, frameon=False, loc="upper right")
        axh.set_ylim(lo, None)
        ax.set_ylabel(r"$\lambda_j$")
    fig.suptitle(r"The floor of $f(y)=y^2e^{-y^2}$: median and 10–90% band over "
                 f"{cfg['n_draws']} sketches", fontsize=10.5, y=0.97)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"fig2_activations_floor_{config}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick", choices=CONFIGS)
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--replot", action="store_true", help="plot from saved floor results")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    R = np.load(os.path.join(args.results, f"fig2_activations_{args.config}.npz"))
    path = os.path.join(args.results, f"fig2_activations_floor_{args.config}.npz")
    if args.replot:
        F = dict(np.load(path))
    else:
        F = compute(cfg, R)
        np.savez_compressed(path, meta=json.dumps(dict(cfg, config=args.config)), **F)
        print(f"saved {path}")
    os.makedirs(args.outdir, exist_ok=True)
    plot(cfg, R, F, args.outdir, args.config)


if __name__ == "__main__":
    main()
