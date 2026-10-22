"""Tables for the documentation page and Table SM1 of the supplement, generated from the results.

Prints (markdown):
  1. slopes per layer, median over seeds with [min, max], fit window 1-100, for every network;
  2. the pre-activation scale per layer of the default-initialization MLPs;
  3. the final-layer slope under several fit windows (plrf.fit.ROBUSTNESS_WINDOWS);
  4. test accuracy of the trained networks.
Writes figures/sm5_table_<config>.tex: Table SM1 (medians over seeds; R^2 is the minimum over layers
and seeds).  Layers whose covariance is carried by a handful of images (top 10 images > 50% of the
trace; only the controlled-scale He3 MLP, layers 3-4) are flagged by run.py's companion check on the
page, not here.

Usage:  python -m experiments.sm5_depth.summarize --config paper
"""

import argparse
import os

import numpy as np

from experiments.sm5_depth.plot import load
from plrf.fit import DEFAULT_WINDOW, ROBUSTNESS_WINDOWS, powerlaw_fit, powerlaw_fit_or_nan

GROUPS = [  # (key, arch, label)
    ("mlpC_relu", "mlp", "MLP, ReLU"), ("mlpC_tanh", "mlp", r"MLP, $\tanh$"), ("mlpC_he3", "mlp", r"MLP, $\mathrm{He}_3$"),
    ("mlpC_relu_rmsnorm", "mlp", "MLP, ReLU + RMSNorm"), ("mlpC_relu_layernorm", "mlp", "MLP, ReLU + LayerNorm"),
    ("mlpD_relu", "mlp", "MLP, ReLU"), ("mlpD_tanh", "mlp", r"MLP, $\tanh$"), ("mlpD_he3", "mlp", r"MLP, $\mathrm{He}_3$"),
    ("mlpD_relu_rmsnorm", "mlp", "MLP, ReLU + RMSNorm"), ("mlpD_relu_layernorm", "mlp", "MLP, ReLU + LayerNorm"),
    ("mlp_trained", "mlp", "MLP, ReLU, trained"),
    ("cnn_random", "cnn", "VGG CNN, random"), ("cnn_trained", "cnn", "VGG CNN, trained"),
    ("resnet_random", "resnet", "ResNet-20, random"), ("resnet_trained", "resnet", "ResNet-20, trained"),
]
SETUP = {"mlpC": "iterated random features (controlled scale)", "mlpD": "default initialization (submitted setup)"}


def fits(R, group, layers, window=DEFAULT_WINDOW):
    """slopes and R^2, each (n_seeds, n_layers)."""
    n = R[f"{group}__{layers[0]}"].shape[0]
    F = np.array([[powerlaw_fit_or_nan(R[f"{group}__{l}"][s], window) for l in layers] for s in range(n)])
    return F[..., 0], F[..., 2]


def label(key, lab):
    p = key.split("_")[0]
    return f"{lab} ({'controlled' if p == 'mlpC' else 'default init'})" if p in SETUP else lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick")
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    R, meta = load(args.results, args.config)
    names = next(iter(meta.values()))["layer_names"]
    s_in, _, r2_in = powerlaw_fit(R["input__centered"][0])
    groups = [g for g in GROUPS if f"{g[0]}__{names[g[1]][0]}" in R]

    print(f"Input: slope {s_in:.2f}, R^2 {r2_in:.3f} (window {DEFAULT_WINDOW})\n")
    print("| network | " + " | ".join(f"layer {i + 1}" for i in range(10)) + " | min $$R^2$$ |")
    for key, arch, lab in groups:
        S, R2 = fits(R, key, names[arch])
        cells = [f"{np.median(S[:, i]):.2f} [{S[:, i].min():.2f}, {S[:, i].max():.2f}]" for i in range(S.shape[1])]
        print(f"| {label(key, lab)} | " + " | ".join(cells) + " |" * (11 - len(cells)) + f" {R2.min():.3f} |")

    if any(k.endswith("__preact_rms") for k in R):
        print("\nPre-activation root mean square per layer, default initialization (median over seeds):\n")
        for key, arch, lab in groups:
            if f"{key}__preact_rms" in R:
                print(f"| {lab} | " + " | ".join(f"{x:.3f}" for x in np.median(R[f'{key}__preact_rms'], 0)) + " |")

    print("\nFinal-layer slope under different fit windows (median over seeds):\n")
    print("| network | " + " | ".join(f"{a}-{b}" for a, b in ROBUSTNESS_WINDOWS) + " |")
    for key, arch, lab in groups:
        last = names[arch][-1:]
        print(f"| {label(key, lab)} | " + " | ".join(f"{np.median(fits(R, key, last, w)[0]):.2f}" for w in ROBUSTNESS_WINDOWS) + " |")
    print(f"| input | " + " | ".join(f"{powerlaw_fit(R['input__centered'][0], w)[0]:.2f}" for w in ROBUSTNESS_WINDOWS) + " |")

    if "trained" in meta:
        print("\nTest accuracy after training (per seed):\n")
        for k, h in meta["trained"]["training_history"].items():
            print(f"  {k}: {h[-1]['test_acc']:.4f}")

    # ---- Table SM1 (LaTeX) ----
    lines = [r"\begin{tabular}{llrrr}", r"\toprule", r"Network & Layer & Dim & Slope & $R^2$ \\", r"\midrule",
             rf"\multicolumn{{2}}{{l}}{{\textit{{Input (CIFAR-10)}}}} & 3072 & ${s_in:.2f}$ & {r2_in:.3f} \\"]
    dims = {"mlp": str(meta.get("mlp", meta.get("trained", {})).get("width", "")), "cnn": "16k--2k"}
    prev = None
    for key, arch, lab in groups:
        sec = key.split("_")[0] if key.startswith("mlp") and key[3] in "CD" else key.split("_")[0]
        if sec != prev:
            lines.append(r"\midrule")
            if sec in SETUP:
                lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{Random MLPs, {SETUP[sec]}}}}} \\")
            prev = sec
        S, R2 = fits(R, key, names[arch])
        med = np.median(S, 0)
        if arch == "resnet":
            groups_r = [(["conv1"], "16384"), (names[arch][1:4], "16384"), (names[arch][4:7], "8192"), (names[arch][7:10], "4096")]
            first = True
            for ls, dim in groups_r:
                idx = [names[arch].index(l) for l in ls]
                lines.append(f"{lab if first else ''} & {' / '.join(ls)} & {dim} & "
                             + " / ".join(f"${med[i]:.2f}$" for i in idx) + f" & {R2[:, idx].min():.3f} \\\\")
                first = False
        else:
            lines.append(f"{lab} & {' / '.join(names[arch])} & {dims.get(arch, '')} & "
                         + " / ".join(f"${m:.2f}$" for m in med) + f" & {R2.min():.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"sm5_table_{args.config}.tex")
    with open(out, "w") as f:
        f.write("% generated by: python -m experiments.sm5_depth.summarize --config " + args.config + "\n")
        f.write(f"% slopes: OLS in log-log over j in [{DEFAULT_WINDOW[0]}, {DEFAULT_WINDOW[1]}], median over seeds\n")
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
