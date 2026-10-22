"""Plot results/fig2_activations_<config>.npz: exact spectra lambda_j(K_W) of four non-polynomial activations,
for each d (light to dark; median over sketches with a 10-90% band).  Rows: pre-activation scale
sigma^2 = 1 and sigma^2 = 3.64.  Dotted grey: j^{-alpha}, through the largest-d median at j = 3.

The floor of y^2 exp(-y^2) is analysed separately in floor.py.

Usage:  python -m experiments.fig2_activations.plot --config quick
Output: figures/fig2_activations_<config>.{pdf,png}
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D_COLORS = ["#86b6ef", "#2a78d6", "#104281"]
TITLES = {"relu": "ReLU", "heaviside": "Heaviside", "tanh": r"$\tanh$", "bump": r"$y^2e^{-y^2}$"}


def band(ax, x, arr, color, q=(10, 50, 90), **kw):
    lo, med, hi = np.percentile(arr, q, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, lw=0)
    ax.plot(x, med, color=color, **kw)
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick")
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--kind", default="K", choices="KC", help="K: second moment (default); C: covariance")
    args = ap.parse_args()
    R = np.load(os.path.join(args.results, f"fig2_activations_{args.config}.npz"))
    meta = json.loads(str(R["meta"]))
    ds, alpha = meta["ds"], meta["alpha"]
    sc = meta["scales"]
    fig, axes = plt.subplots(len(sc), len(TITLES), figsize=(15, 4 * len(sc)), gridspec_kw=dict(hspace=0.35, wspace=0.25))
    for r, (sname, sig2) in enumerate(sc.items()):
        for c, a in enumerate(TITLES):
            ax = axes[r, c]
            for d, col in zip(ds, D_COLORS[-len(ds):]):
                med = band(ax, np.arange(1, d + 1), R[f"{sname}__{a}__d{d}__{args.kind}"], col, lw=1.5, label=f"$d={d}$")
            j = np.arange(1, max(ds) + 1)
            ax.plot(j, med[2] * (j / 3.0) ** (-alpha), color="0.55", ls=":", lw=1.2, label=rf"$j^{{-{alpha}}}$")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(alpha=0.2, which="both")
            ax.set_title(f"{TITLES[a]},  $\\sigma^2={sig2:.2f}$", fontsize=11)
            if r == len(sc) - 1:
                ax.set_xlabel("$j$")
        axes[r, 0].set_ylabel(r"$\lambda_j(K_W)$")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
    what = "second moment $K_W$" if args.kind == "K" else "covariance"
    fig.suptitle(rf"Gaussian data, $H\propto\mathrm{{diag}}(j^{{-{alpha}}})$, pre-activations of mean square $\sigma^2$, "
                 rf"$v=2d$; exact {what}; median and 10–90% band over {meta['n_draws']} sketches", fontsize=10.5, y=0.97)
    os.makedirs(args.outdir, exist_ok=True)
    suffix = args.config + ("" if args.kind == "K" else "_centered")
    for ext in ("pdf", "png"):
        out = os.path.join(args.outdir, f"fig2_activations_{suffix}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
