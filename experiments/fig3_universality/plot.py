"""Plot results/fig3_universality_<config>.npz.

Two figures, each with one column per activation (y^2, y^3, ReLU, tanh, y^2 exp(-y^2)):

fig3_universality_synthetic_<config>: inputs x = H^{1/2} xi with i.i.d. standardized coordinates.
  Top: spectra lambda_j(K_W), median over sketches with a 10-90% band.  Gaussian in black;
  exact where a closed form exists (solid black), otherwise Monte Carlo (solid) with the exact
  Gaussian dashed where available.
  Bottom: lambda_j(K_W; distribution) / lambda_j(K_W; Gaussian), computed per sketch (same W)
  and then summarized; exact/exact for y^2, y^3 and Monte Carlo/Monte Carlo otherwise.
fig3_universality_cifar_<config>: CIFAR-10 against a Gaussian with the same covariance.
  Both at the same sample size (the 60,000 images; 60,000 Gaussian samples); the exact
  Gaussian spectrum is dashed where available.
fig3_input_spectra_<config>: the input spectra, H and the CIFAR-10 covariance.

All matrices are second moments K_W (the theorem's object); the covariances are stored as well
(see summarize.py).

Usage:  python -m experiments.fig3_universality.plot --config quick
Output: figures/fig3_*_<config>.{pdf,png}
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

INK = "#1f1f1e"
STYLE = {  # distribution: (color, linestyle, label)
    "gauss": (INK, "-", "Gaussian"),
    "rad": ("#2a78d6", "-", "Rademacher"),
    "t9": ("#eb6834", "-", r"Student-$t_9$"),
    "t3": ("#1baf7a", (0, (4, 1.5)), r"Student-$t_3$"),
    "cifar": ("#2a78d6", "-", "CIFAR-10"),
    "gsur": (INK, "-", "Gaussian, same cov."),
}
TITLES = {"y2": r"$f(y)=y^2$", "y3": r"$f(y)=y^3$", "relu": "ReLU", "tanh": r"$\tanh$", "bump": r"$y^2e^{-y^2}$"}
ACTS = list(TITLES)


def band(ax, x, arr, color, ls="-", lw=1.5, label=None, q=(10, 50, 90)):
    lo, med, hi = np.percentile(arr, q, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(x, med, color=color, ls=ls, lw=lw, label=label)


def ratio_axis(ax):
    ax.axhline(1, color="0.4", lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0.2, 5)
    ax.set_yticks([0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(["1/4", "1/2", "1", "2", "4"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("$j$")
    ax.grid(alpha=0.2, which="both")


def spectra_row(axes, R, kind, rows, ref, j, tag):
    """One row of panels (one per activation): the reference spectrum (Gaussian) and the other inputs,
    median over sketches with a 10-90% band; dashed: the Gaussian closed form where it exists."""
    for ax, a in zip(axes, ACTS):
        rdist, rmeth, rexact = ref[a]
        col, ls, lab = STYLE[rdist]
        band(ax, j, R[f"{a}__{rdist}__{rmeth}__{kind}"], col, ls, label=lab)
        if rexact:
            ax.plot(j, np.median(R[f"{a}__{rdist}__exact__{kind}"], 0), color=INK, ls=(0, (2, 2)), lw=1.1,
                    label="closed form")
        for dist, meth in rows[a]:
            col, ls, lab = STYLE[dist]
            band(ax, j, R[f"{a}__{dist}__{meth}__{kind}"], col, ls, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.grid(alpha=0.2, which="both")
        ax.text(0.97, 0.95, tag(a), transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.35")
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"saved {out}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick")
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--kind", default="K", choices="KC", help="K: second moment (default); C: covariance")
    args = ap.parse_args()
    F = np.load(os.path.join(args.results, f"fig3_universality_{args.config}.npz"))
    meta = json.loads(str(F["meta"]))
    d, nd = meta["d"], meta["n_draws"]
    j = np.arange(1, d + 1)
    suffix = args.config + ("" if args.kind == "K" else "_centered")
    what = "second moment $K_W$" if args.kind == "K" else "covariance"

    syn = {k[len("syn__"):]: F[k] for k in F.files if k.startswith("syn__")}
    cif = {k[len("cifar__"):]: F[k] for k in F.files if k.startswith("cifar__")}
    fig, axes = plt.subplots(2, len(ACTS), figsize=(17, 7.6), gridspec_kw=dict(hspace=0.32, wspace=0.28))
    rows = {a: [("rad", "exact"), ("t9", "exact")] for a in ("y2", "y3")}
    rows.update({"relu": [("rad", "mc"), ("t9", "mc")], "tanh": [("rad", "mc"), ("t9", "mc"), ("t3", "mc")],
                 "bump": [("rad", "mc"), ("t9", "mc"), ("t3", "mc")]})
    ref = {"y2": ("gauss", "exact", False), "y3": ("gauss", "exact", False), "relu": ("gauss", "mc", True),
           "tanh": ("gauss", "mc", False), "bump": ("gauss", "mc", False)}
    spectra_row(axes[0], syn, args.kind, rows, ref, j,
                lambda a: "exact" if a in ("y2", "y3") else f"Monte Carlo, $m={meta['m']}$")
    rows = {a: [("cifar", "data")] for a in ACTS}
    ref = {a: ("gsur", "mc", a in ("y2", "y3", "relu")) for a in ACTS}
    spectra_row(axes[1], cif, args.kind, rows, ref, j, lambda a: f"{meta['cifar_N']} samples each")
    for ax, a in zip(axes[0], ACTS):
        ax.set_title(TITLES[a], fontsize=12)
    for ax in axes[1]:
        ax.set_xlabel("$j$")
    axes[0, 0].set_ylabel(r"$\lambda_j(K_W)$" if args.kind == "K" else r"$\lambda_j$(covariance)")
    axes[1, 0].set_ylabel(r"$\lambda_j(K_W)$" if args.kind == "K" else r"$\lambda_j$(covariance)")
    axes[0, 0].text(-0.3, 1.13, rf"(a) synthetic: $x=H^{{1/2}}\xi$, $\xi_k$ i.i.d., $H\propto\mathrm{{diag}}(j^{{-{meta['alpha']}}})$, "
                    rf"$\mathrm{{tr}}\,H=1$, $v={meta['v']}$", transform=axes[0, 0].transAxes, fontsize=10.5)
    axes[1, 0].text(-0.3, 1.06, rf"(b) CIFAR-10 (60,000 images, centered, $v={meta['cifar_v']}$) and a Gaussian with the same covariance",
                    transform=axes[1, 0].transAxes, fontsize=10.5)
    fig.suptitle(f"{what}, $d={d}$; median and 10–90% band over {nd} sketches $W$ (the same $W$ for every input)",
                 fontsize=10.5, y=0.99)
    save(fig, args.outdir, f"fig3_universality_{suffix}")

    if args.kind == "K":
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        H, h = F["H"], F["cifar_spectrum"]
        ax.plot(np.arange(1, len(H) + 1), H, color=INK, lw=1.5, label=rf"synthetic $H$, $\alpha={meta['alpha']}$")
        ax.plot(np.arange(1, len(h) + 1), h, color=STYLE["cifar"][0], lw=1.5, label="CIFAR-10 covariance")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$j$"); ax.set_ylabel(r"$\lambda_j(\mathrm{Cov}\,x)$ (unit trace)")
        ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2, which="both")
        save(fig, args.outdir, f"fig3_input_spectra_{args.config}")


if __name__ == "__main__":
    main()
