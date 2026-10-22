"""Plot results/sm6_alpha_<config>.npz.

sm6_principal_<config>: the principal products H_{s_1}...H_{s_p} for p = 1..4 against the theorem's
  form (log^{p-1}(j+1)/j)^alpha.
sm6_reproduce_<config>: the original figure's computation (alpha = 0.7, one sketch, Monte Carlo sample
  covariance with m = d samples, not divided by d), with d H^{1_p} (the sorted products, not divided by d)
  as solid lines, on the same axes as the original.
sm6_exact_<config>: rows alpha = 0.7, 0.5; columns p = 1..4.  The exact spectrum of K_W for each d
  (light to dark; median over sketches with a 10-90% band), divided by the largest eigenvalue of the
  population spectrum at that d (p <= 3), or by its own median lambda_1 (p = 4, no population spectrum).
  Dashed: the population spectrum (p <= 3).  Dotted grey: j^{-alpha}.

Usage:  python -m experiments.sm6_alpha.plot --config quick
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P_COLORS = {1: "#1f1f1e", 2: "#2a78d6", 3: "#eb6834", 4: "#1baf7a"}
D_COLORS = ["#86b6ef", "#2a78d6", "#104281"]


def band(ax, x, arr, color, q=(10, 50, 90), **kw):
    lo, med, hi = np.percentile(arr, q, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2, lw=0)
    ax.plot(x, med, color=color, **kw)
    return med


def reproduce_figure(R, meta):
    r = meta["reproduce"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for p in meta["ps"]:
        if f"repro_p{p}" not in R:
            continue
        lam = R[f"repro_p{p}"]
        j = np.arange(1, len(lam) + 1)
        ax.plot(j[lam > 0], lam[lam > 0], ".", ms=2.5, color=P_COLORS[p], label=f"$y^{p}$, sample covariance")
        pr = R[f"repro_principal_p{p}"]
        ax.plot(np.arange(1, len(pr) + 1), pr, "-", lw=1.2, color=P_COLORS[p], alpha=0.8,
                label=rf"$d\,H^{{\mathbf{{1}}_{p}}}$ (principal products)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-3, None)
    ax.set_xlabel("eigenvalue index $j$"); ax.set_ylabel("$j$-th eigenvalue")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower left")
    ax.grid(alpha=0.2, which="both")
    ax.set_title(rf"Reproduction of the original computation: $\alpha={r['alpha']}$, $d={r['d']}$, $m={r['m']}$ Monte Carlo "
                 rf"samples, $v=d^{{1/p+{meta['eps']}}}$, one sketch", fontsize=10)
    return fig


def principal_figure(R, meta):
    """The principal products H_{s_1}...H_{s_p} (s_1 < ... < s_p <= v) in decreasing order, for p = 1..4 at
    the largest d (v = d^{1/p + eps}), each divided by its largest value.  Dashed: the theorem's form
    (log^{p-1}(j+1)/j)^alpha, scaled to agree at j = 100.  The legend gives local slopes over j in [100, 1000]."""
    from plrf.fit import powerlaw_fit
    d = max(meta["ds"])
    alphas = sorted(meta["alphas"], reverse=True)
    fig, axes = plt.subplots(1, len(alphas), figsize=(6 * len(alphas), 4.6), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, a in zip(axes, alphas):
        for p in meta["ps"]:
            pr = R[f"principal_a{a}_p{p}_d{d}"]
            pr = pr / pr[0]
            j = np.arange(1, len(pr) + 1)
            th = (np.log(j + 1) ** (p - 1) / j) ** a
            th = th * pr[99] / th[99]
            s_pr, s_th = powerlaw_fit(pr, (100, 1000))[0], powerlaw_fit(th, (100, 1000))[0]
            v = int(round(d ** (1 / p + meta["eps"])))
            ax.plot(j, pr, color=P_COLORS[p], lw=1.7, label=f"$p={p}$, $v={v}$: slope {s_pr:.2f} (form {s_th:.2f})")
            ax.plot(j, th, color=P_COLORS[p], ls="--", lw=1)
        ax.plot(j, j ** (-a), color="0.6", ls=":", lw=1.2, label=rf"$j^{{-{a}}}$")
        ax.plot([], [], "k--", lw=1, label=r"$(\log^{p-1}(j+1)/j)^{\alpha}$, matched at $j=100$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(alpha=0.2, which="both"); ax.set_xlabel("$j$")
        ax.set_title(rf"$\alpha={a}$", fontsize=11)
        ax.legend(fontsize=7.8, frameon=False, loc="lower left")
    axes[0].set_ylabel(r"$j$-th largest product $H_{s_1}\cdots H_{s_p}$ (divided by the largest)")
    fig.suptitle(rf"Principal products for $H=\mathrm{{diag}}(j^{{-\alpha}})$, $s_1<\dots<s_p\leq v=d^{{1/p+{meta['eps']}}}$, "
                 rf"$d={d}$ (legend: local slopes over $100\leq j\leq 1000$)", fontsize=10.5, y=0.99)
    return fig


def exact_figure(R, meta):
    ps, ds, alphas = meta["ps"], meta["ds"], sorted(meta["alphas"], reverse=True)
    fig, axes = plt.subplots(len(alphas), len(ps), figsize=(15, 4.1 * len(alphas)), sharey="row",
                             gridspec_kw=dict(hspace=0.3, wspace=0.08))
    for r, a in enumerate(alphas):
        for c, p in enumerate(ps):
            ax = axes[r, c]
            for d, col in zip(ds, D_COLORS[-len(ds):]):
                key = f"a{a}_p{p}_d{d}"
                lam = R[f"lam_{key}"]
                pop = R.get(f"pop_{key}") if f"pop_{key}" in R.files else None
                c0 = pop[0] if pop is not None else np.median(lam[:, 0])
                j = np.arange(1, d + 1)
                band(ax, j, lam / c0, col, lw=1.5, label=f"$d={d}$, $v={int(round(d ** (1 / p + meta['eps'])))}$")
                if pop is not None:
                    ax.plot(j, pop[:d] / c0, color=col, ls="--", lw=0.9)
            j = np.arange(1, max(ds) + 1)
            ax.plot(j, j ** (-a), color="0.6", ls=":", lw=1.2, label=rf"$j^{{-{a}}}$")
            if p <= 3:
                ax.plot([], [], "k--", lw=0.9, label="population spectrum")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_ylim(1e-4, 2)
            ax.grid(alpha=0.2, which="both")
            ax.set_title(rf"$\alpha={a}$, $f(y)=y^{p}$" + ("" if p <= 3 else " (no population spectrum)"), fontsize=10)
            ax.legend(fontsize=7, frameon=False, loc="lower left")
            if r == len(alphas) - 1:
                ax.set_xlabel("$j$")
        axes[r, 0].set_ylabel(r"$\lambda_j(K_W)$ / $\lambda_1$(population)")
    fig.suptitle(rf"Exact spectra for $\alpha<1$: $H=\mathrm{{diag}}(j^{{-\alpha}})$, $v=d^{{1/p+{meta['eps']}}}$; median and 10–90% band "
                 rf"over {meta['n_draws']} sketches", fontsize=10.5, y=0.96)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick")
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    R = np.load(os.path.join(args.results, f"sm6_alpha_{args.config}.npz"))
    meta = json.loads(str(R["meta"]))
    os.makedirs(args.outdir, exist_ok=True)
    figs = [("sm6_principal", principal_figure(R, meta)), ("sm6_exact", exact_figure(R, meta))]
    if "repro_p1" in R.files:
        figs.append(("sm6_reproduce", reproduce_figure(R, meta)))     # shown on the revision-notes page
    for name, fig in figs:
        for ext in ("pdf", "png"):
            out = os.path.join(args.outdir, f"{name}_{args.config}.{ext}")
            fig.savefig(out, bbox_inches="tight", dpi=150)
            print(f"saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
