"""The staircase for alpha < 1: compute and plot.

In f(y) = y^p = sum_k b_k s^{(p-k)/2} :y^k:, the Wick component of degree k carries the weight
s^{p-k} ~ (tr H)^{p-k}.  For alpha > 1, tr H is bounded; for alpha < 1 it grows like v^{1-alpha}, so the
low-degree components dominate.  The degree-k component supplies about C(v, k) eigenvalues (one per
squarefree index set, plus O(v^{k-1}) from repeated indices), after which it runs out and the spectrum
drops to the next component: a staircase, with steps near j = v, v + C(v, 2), ...

Two panels, alpha in {1.31, 0.7, 0.5}, H = diag(j^{-alpha}) (not normalized):
  (a) f = y^3, v = 341 (= round(10^4^{1/3 + 0.3}), the largest v of the main figure): the exact population
      spectrum (top d = 10^4 eigenvalues).  Steps at j = v.
  (b) f = y^4, v = 30, d = 3000: the exact spectrum of K_W, median over n_draws sketches (no population
      spectrum is implemented for p = 4).  Steps at j = 1 (the mean) and j = v + C(v, 2) = 465 (the end of
      the degree-2 component).

Usage:  python -m experiments.sm6_alpha.staircase --config paper
Output: results/sm6_staircase_<config>.npz, figures/sm6_staircase_<config>.{pdf,png}
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.fig2_theorem.run import spectra_numpy
from plrf.activations import monomial
from plrf.exact import power_law_spectrum
from plrf.population import population_spectrum_poly

CONFIGS = {"quick": dict(alphas=[1.31, 0.7, 0.5], n_draws=3, v3=341, n3=10000, v4=30, d4=1000),
           "paper": dict(alphas=[1.31, 0.7, 0.5], n_draws=20, v3=341, n3=10000, v4=30, d4=3000)}
A_COLORS = {1.31: "#1f1f1e", 0.7: "#2a78d6", 0.5: "#eb6834"}


def compute(cfg):
    out = {}
    for a in cfg["alphas"]:
        H = power_law_spectrum(cfg["v3"], a)
        out[f"pop3_a{a}"] = population_spectrum_poly(monomial(3), H, n=cfg["n3"], head=cfg["v3"])
        H4 = power_law_spectrum(cfg["v4"], a)
        out[f"lam4_a{a}"] = np.array([spectra_numpy(
            np.random.default_rng([6, 2, a == 0.5, a == 0.7, draw]).standard_normal((cfg["v4"], cfg["d4"])), H4, [monomial(4)])[0]
            for draw in range(cfg["n_draws"])])
        print(f"alpha={a}: done", flush=True)
    return out


def plot(cfg, F, outdir, config):
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(wspace=0.22))
    for a in cfg["alphas"]:
        pop = F[f"pop3_a{a}"]
        ax3.plot(np.arange(1, len(pop) + 1), pop / pop[0], color=A_COLORS[a], lw=1.6,
                 label=rf"$\alpha={a}$, $\mathrm{{tr}}\,H={power_law_spectrum(cfg['v3'], a).sum():.1f}$")
        lam = F[f"lam4_a{a}"]
        lo, med, hi = np.percentile(lam / np.median(lam[:, 0]), (10, 50, 90), axis=0)
        j = np.arange(1, lam.shape[1] + 1)
        ax4.fill_between(j, lo, hi, color=A_COLORS[a], alpha=0.2, lw=0)
        ax4.plot(j, med, color=A_COLORS[a], lw=1.6, label=rf"$\alpha={a}$")
    v3, v4 = cfg["v3"], cfg["v4"]
    ax3.axvline(v3, color="0.5", ls=":", lw=1)
    ax3.text(v3 * 1.08, 0.3, f"$j=v={v3}$", fontsize=8.5, color="0.35")
    k2 = v4 + v4 * (v4 - 1) // 2
    for x, t in [(1, None), (k2, rf"$j=v+\binom{{v}}{{2}}={k2}$")]:
        if t:
            ax4.axvline(x, color="0.5", ls=":", lw=1)
            ax4.text(x * 1.08, 0.3, t, fontsize=8.5, color="0.35")
    for ax, t in [(ax3, rf"(a) $f(y)=y^3$, $v={v3}$: population spectrum"),
                  (ax4, rf"(b) $f(y)=y^4$, $v={v4}$, $d={cfg['d4']}$: exact $K_W$, median over {cfg['n_draws']} sketches")]:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$j$"); ax.grid(alpha=0.2, which="both")
        ax.set_title(t, fontsize=10, loc="left")
        ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax3.set_ylabel(r"$\lambda_j / \lambda_1$")
    fig.suptitle(r"The staircase: $H=\mathrm{diag}(j^{-\alpha})$; steps where a Wick component runs out, growing with $\mathrm{tr}\,H$",
                 fontsize=10.5, y=1.0)
    for ext in ("pdf", "png"):
        p = os.path.join(outdir, f"sm6_staircase_{config}.{ext}")
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick", choices=CONFIGS)
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--replot", action="store_true")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    path = os.path.join(args.results, f"sm6_staircase_{args.config}.npz")
    if args.replot:
        F = dict(np.load(path))
    else:
        F = compute(cfg)
        np.savez_compressed(path, meta=json.dumps(dict(cfg, config=args.config,
                            seed_rule="y^4 sketches: default_rng([6, 2, alpha==0.5, alpha==0.7, draw])")), **F)
    os.makedirs(args.outdir, exist_ok=True)
    plot(cfg, F, args.outdir, args.config)


if __name__ == "__main__":
    main()
