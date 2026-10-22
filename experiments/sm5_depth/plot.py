"""Plot results/sm5_depth_{mlp,conv,trained}_<config>.npz.

sm5_mlp_slopes_<config>: fitted slope (window 1-100) against layer for the random MLPs.  Columns:
  controlled scale (iterated random features) and the submitted default initialization; rows:
  activations (ReLU, tanh, He3) and normalizations (ReLU, +RMSNorm, +LayerNorm).  Median over seeds,
  bars from the minimum to the maximum.  Dashed: the input slope.
sm5_slopes_<config>: slope against relative depth for the CNN, ResNet-20 and ReLU MLP (setup D),
  random (dashed) and trained (solid).
sm5_spectra_<config>: lambda_j / tr at every layer (light to dark with depth) and the input (black),
  median over seeds, for each network.

Usage:  python -m experiments.sm5_depth.plot --config quick
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plrf.fit import DEFAULT_WINDOW, powerlaw_fit, powerlaw_fit_or_nan

INK = "#1f1f1e"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]           # categorical slots 1-3
DEPTH = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281", "#0d366b"]
ARCH = {"mlp": ("ReLU MLP", "#2a78d6"), "cnn": ("VGG CNN", "#eb6834"), "resnet": ("ResNet-20", "#1baf7a")}


def load(results, config):
    R, meta = {}, {}
    for part in ("mlp", "conv", "trained"):
        p = os.path.join(results, f"sm5_depth_{part}_{config}.npz")
        if os.path.exists(p):
            F = np.load(p)
            R.update({k: F[k] for k in F.files if k != "meta"})
            meta[part] = json.loads(str(F["meta"]))
    return R, meta


def slopes(R, group, layers, window=DEFAULT_WINDOW):
    """(n_seeds, n_layers) array of fitted slopes."""
    return np.array([[powerlaw_fit_or_nan(R[f"{group}__{l}"][s], window)[0] for l in layers]
                     for s in range(R[f"{group}__{layers[0]}"].shape[0])])


def errplot(ax, x, S, color, ls="-", label=None, marker="o"):
    med = np.median(S, 0)
    ax.errorbar(x, med, yerr=[med - S.min(0), S.max(0) - med], color=color, ls=ls, marker=marker, ms=5,
                lw=1.6, capsize=3, label=label)


def mlp_figure(R, meta):
    layers = meta["mlp"]["layer_names"]["mlp"]
    s_in = powerlaw_fit(R["input__centered"][0])[0]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharey="row", gridspec_kw=dict(hspace=0.35, wspace=0.08))
    rows = [[("relu", "ReLU"), ("tanh", r"$\tanh$")],
            [("relu", "ReLU"), ("relu_rmsnorm", "ReLU + RMSNorm"), ("relu_layernorm", "ReLU + LayerNorm")]]
    cols = [("mlpC", "iterated random features, scale controlled"),
            ("mlpD", "submitted setup: default init, biases")]
    x = np.arange(len(layers) + 1)
    for r, row in enumerate(rows):
        for c, (prefix, title) in enumerate(cols):
            ax = axes[r, c]
            for (v, lab), col in zip(row, CAT):
                S = np.c_[np.full(R[f"{prefix}_{v}__L1"].shape[0], s_in), slopes(R, f"{prefix}_{v}", layers)]
                errplot(ax, x, S, col, label=lab)
            ax.axhline(s_in, color="0.5", ls="--", lw=1)
            ax.set_xticks(x); ax.set_xticklabels(["input"] + layers)
            ax.grid(alpha=0.25)
            ax.set_title(f"({'ab'[c] if r == 0 else 'cd'[c]}) {title}", fontsize=9.5, loc="left")
            if c == 0:
                ax.set_ylabel(f"fitted slope, $j\\in[{DEFAULT_WINDOW[0]},{DEFAULT_WINDOW[1]}]$")
            ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    n = R["mlpC_relu__L1"].shape[0]
    fig.suptitle(f"Random-initialization MLPs on CIFAR-10 (width {meta['mlp']['width']}); median and range over {n} seeds; "
                 f"dashed: input slope {s_in:.2f}", fontsize=10.5, y=0.97)
    return fig


def he3_figure(R, meta):
    """He_3 = y^3 - 3y in the two MLP setups: slopes per layer, and the controlled-scale spectra."""
    layers = meta["mlp"]["layer_names"]["mlp"]
    s_in = powerlaw_fit(R["input__centered"][0])[0]
    fig, (ax, axs) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(wspace=0.25))
    x = np.arange(len(layers) + 1)
    for (prefix, lab), col in zip([("mlpC", "iterated random features, scale controlled"),
                                   ("mlpD", "submitted setup: default init, biases")], CAT):
        S = np.c_[np.full(R[f"{prefix}_he3__L1"].shape[0], s_in), slopes(R, f"{prefix}_he3", layers)]
        errplot(ax, x, S, col, label=lab)
    ax.axhline(s_in, color="0.5", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(["input"] + layers)
    ax.set_ylabel(f"fitted slope, $j\\in[{DEFAULT_WINDOW[0]},{DEFAULT_WINDOW[1]}]$")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.grid(alpha=0.25)
    ax.set_title(r"(a) $\mathrm{He}_3(y)=y^3-3y$ MLP: fitted slope per layer", fontsize=10, loc="left")
    lin = R["input__centered"][0]
    axs.plot(np.arange(1, len(lin) + 1), lin / lin.sum(), color=INK, lw=1.6, label="input")
    for l, col in zip(layers, DEPTH[-len(layers):]):
        lam = np.median(R[f"mlpC_he3__{l}"], 0)
        lam = lam[lam > 0]
        axs.plot(np.arange(1, len(lam) + 1), lam / lam.sum(), color=col, lw=1.2, label=l)
    axs.set_xscale("log"); axs.set_yscale("log"); axs.set_ylim(1e-9, 1)
    axs.axvspan(*DEFAULT_WINDOW, color="0.9", zorder=0)
    axs.grid(alpha=0.2, which="both"); axs.set_xlabel("$j$"); axs.set_ylabel(r"$\lambda_j/\mathrm{tr}$")
    axs.legend(fontsize=8, frameon=False, loc="lower left")
    axs.set_title("(b) spectra per layer, scale controlled (median over seeds)", fontsize=10, loc="left")
    return fig


def fits_figure(R, meta):
    """Spectra of the input and of the first, a middle and the last layer, with the fitted lines."""
    names = meta.get("conv", meta.get("trained", meta.get("mlp")))["layer_names"]
    panels = [("mlpC_relu", "mlp", "ReLU MLP, iterated RF"), ("cnn_random", "cnn", "VGG CNN, random"),
              ("resnet_random", "resnet", "ResNet-20, random"), ("mlp_trained", "mlp", "ReLU MLP, trained"),
              ("cnn_trained", "cnn", "VGG CNN, trained"), ("resnet_trained", "resnet", "ResNet-20, trained")]
    panels = [p for p in panels if f"{p[0]}__{names[p[1]][0]}" in R]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), gridspec_kw=dict(hspace=0.3, wspace=0.22))
    a, b = DEFAULT_WINDOW
    for ax, (group, arch, title) in zip(axes.flat, panels):
        layers = names[arch]
        pick = [("input", None), (layers[0], None), (layers[len(layers) // 2], None), (layers[-1], None)]
        for (l, _), col in zip(pick, [INK] + CAT):
            lam = R["input__centered"][0] if l == "input" else np.median(R[f"{group}__{l}"], 0)
            lam = lam[lam > 0] / lam[lam > 0].sum()
            j = np.arange(1, len(lam) + 1)
            slope, icpt, _ = powerlaw_fit(lam, DEFAULT_WINDOW)
            ax.plot(j, lam, color=col, lw=1.3, label=f"{l}: slope {slope:.2f}")
            jj = np.arange(a, b + 1)
            ax.plot(jj, 10**icpt * jj**slope, color=col, ls="--", lw=1.6, alpha=0.9)
            jx = np.arange(b, 1001)
            ax.plot(jx, 10**icpt * jx**slope, color=col, ls=":", lw=0.9, alpha=0.6)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-7, 1)
        ax.axvspan(a, b, color="0.92", zorder=0)
        ax.grid(alpha=0.2, which="both"); ax.set_xlabel("$j$")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\lambda_j / \mathrm{tr}$")
    fig.suptitle(f"Fitted power laws (dashed: OLS fit over $j\\in[{a},{b}]$, shaded; dotted: its extrapolation); "
                 "median spectra over seeds", fontsize=10.5, y=0.95)
    return fig


def depth_figure(R, meta):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    s_in = powerlaw_fit(R["input__centered"][0])[0]
    names = meta.get("conv", meta.get("trained", meta.get("mlp")))["layer_names"]
    for arch, (lab, col) in ARCH.items():
        layers = names[arch]
        x = np.r_[0, np.arange(1, len(layers) + 1) / len(layers)]
        for state, ls, mk in [("random", "--", "o"), ("trained", "-", "s")]:
            group = ("mlpD_relu" if arch == "mlp" else f"{arch}_random") if state == "random" else f"{arch}_trained"
            if f"{group}__{layers[0]}" not in R:
                continue
            S = np.c_[np.full(R[f"{group}__{layers[0]}"].shape[0], s_in), slopes(R, group, layers)]
            errplot(ax, x, S, col, ls=ls, marker=mk, label=f"{lab}, {state}")
    ax.axhline(s_in, color="0.5", ls="--", lw=1)
    ax.set_xlabel("relative depth (0 = input)")
    ax.set_ylabel(f"fitted slope, $j\\in[{DEFAULT_WINDOW[0]},{DEFAULT_WINDOW[1]}]$")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower left")
    ax.grid(alpha=0.25)
    ax.set_title("Slope across layers on CIFAR-10 (median and range over seeds)", fontsize=10)
    return fig


def spectra_figure(R, meta):
    names = meta.get("conv", meta.get("trained", meta.get("mlp")))["layer_names"]
    panels = [("mlpC_relu", "mlp", "ReLU MLP, iterated RF"), ("mlpC_tanh", "mlp", r"$\tanh$ MLP, iterated RF"),
              ("mlpC_he3", "mlp", r"$\mathrm{He}_3$ MLP, iterated RF"), ("cnn_random", "cnn", "VGG CNN, random"),
              ("resnet_random", "resnet", "ResNet-20, random"), ("mlp_trained", "mlp", "ReLU MLP, trained"),
              ("cnn_trained", "cnn", "VGG CNN, trained"), ("resnet_trained", "resnet", "ResNet-20, trained")]
    panels = [p for p in panels if f"{p[0]}__{names[p[1]][0]}" in R]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), gridspec_kw=dict(hspace=0.3, wspace=0.25))
    lin = R["input__centered"][0]
    for ax, (group, arch, title) in zip(axes.flat, panels):
        ax.plot(np.arange(1, len(lin) + 1), lin / lin.sum(), color=INK, lw=1.6, label="input")
        layers = names[arch]
        cols = DEPTH[-len(layers):] if len(layers) <= len(DEPTH) else plt.cm.Blues(np.linspace(0.35, 1, len(layers)))
        for l, col in zip(layers, cols):
            lam = np.median(R[f"{group}__{l}"], 0)
            lam = lam[lam > 0]
            ax.plot(np.arange(1, len(lam) + 1), lam / lam.sum(), color=col, lw=1.1, label=l)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(1e-7, 1)
        ax.axvspan(*DEFAULT_WINDOW, color="0.9", zorder=0)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2, which="both")
        ax.set_xlabel("$j$")
        ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel(r"$\lambda_j / \mathrm{tr}$")
    axes[1, 0].set_ylabel(r"$\lambda_j / \mathrm{tr}$")
    fig.suptitle("Covariance spectra of CIFAR-10 features at every layer (median over seeds; shaded: fit window)",
                 fontsize=10.5, y=0.95)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick")
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    R, meta = load(args.results, args.config)
    os.makedirs(args.outdir, exist_ok=True)
    figs = []
    if "mlp" in meta:
        figs.append(("sm5_mlp_slopes", mlp_figure(R, meta)))
        figs.append(("sm5_he3", he3_figure(R, meta)))
    figs.append(("sm5_fits", fits_figure(R, meta)))
    figs.append(("sm5_slopes", depth_figure(R, meta)))
    figs.append(("sm5_spectra", spectra_figure(R, meta)))
    for name, fig in figs:
        for ext in ("pdf", "png"):
            out = os.path.join(args.outdir, f"{name}_{args.config}.{ext}")
            fig.savefig(out, bbox_inches="tight", dpi=150)
            print(f"saved {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
