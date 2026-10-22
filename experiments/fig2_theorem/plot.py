"""Plot results/fig2_theorem_<config>.npz: exact spectra lambda_j(K_W) at the largest d (median over
sketches with a 10-90% band), each divided by the largest population eigenvalue of its activation (a
deterministic constant), against the population spectrum (dashed).

(a) monomials y, y^2 (also centered, dotted), y^3;  (b) non-monomial polynomials y + y^2/2 + y^3/6 and
He_3(y) = y^3 - 3y.  Dotted grey: j^{-alpha}.

Usage:  python -m experiments.fig2_theorem.plot --config quick
Output: figures/fig2_theorem_<config>.{pdf,png}
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"y1": "#2a78d6", "y2": "#eb6834", "y2c": "#eb6834", "y3": "#1baf7a", "cub": "#2a78d6", "he3": "#eb6834"}
LABELS = {"y1": r"$y$", "y2": r"$y^2$", "y2c": r"$y^2$, centered", "y3": r"$y^3$",
          "cub": r"$y + y^2/2 + y^3/6$", "he3": r"$\mathrm{He}_3(y) = y^3 - 3y$"}
PANELS = [("(a) monomials", ["y1", "y2", "y3"]), ("(b) polynomials that are not monomials", ["cub", "he3"])]


def band(ax, x, arr, color, q=(10, 50, 90), **kw):
    lo, med, hi = np.percentile(arr, q, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, lw=0)
    return ax.plot(x, med, color=color, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick")
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    R = np.load(os.path.join(args.results, f"fig2_theorem_{args.config}.npz"))
    meta = json.loads(str(R["meta"]))
    alpha, dmax = meta["alpha"], max(meta["ds"])
    j = np.arange(1, dmax + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, (title, keys) in zip(axes, PANELS):
        for key in keys:
            if f"lam_{key}_d{dmax}" not in R:
                continue
            lam, pop = R[f"lam_{key}_d{dmax}"], R[f"pop_{key}_d{dmax}"]
            c = pop[0]
            band(ax, j, lam / c, COLORS[key], lw=1.8, ls=":" if key == "y2c" else "-", label=LABELS[key])
            ax.plot(j, pop / c, color="k", ls="--", lw=0.9)
        ax.plot(j, j ** (-alpha), color="0.6", ls=":", lw=1.2, label=rf"$j^{{-{alpha}}}$")
        ax.plot([], [], "k--", lw=0.9, label="population spectrum")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("eigenvalue index $j$")
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.legend(fontsize=8.5, frameon=False, loc="lower left")
        ax.grid(alpha=0.2, which="both")
    axes[0].set_ylabel(r"$\lambda_j(K_W)\,/\,\lambda_1(\mathrm{population})$")
    fig.suptitle(rf"Exact spectra of $K_W$: Gaussian data, $H\propto\mathrm{{diag}}(j^{{-{alpha}}})$, "
                 rf"$\mathrm{{tr}}\,H=1$, $d={dmax}$, $v=2d$; median and 10–90% band over {meta['n_draws']} sketches",
                 fontsize=10.5, y=0.99)
    os.makedirs(args.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(args.outdir, f"fig2_theorem_{args.config}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
