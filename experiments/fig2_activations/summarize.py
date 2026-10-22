"""Print the summary tables quoted on the documentation page.

1. Convergence in d of the head: median over sketches of lambda_j(K_W at d) / lambda_j(K_W at d_max)
   at a few j.
2. The floor.  d * lambda_j at j = d/2 (median over sketches) for each d: the same value for every d
   means a floor of height proportional to 1/d.  Next to it, the Hermite energy of the activation,
   E_k = mean_i a_k(s_i)^2 (so sum_k E_k = mean_i E f(y_i)^2), grouped by degree.  The part of K_W in
   degree k is (1/d) a_k a_k^T o rho^k; for large k it is close to (1/d) diag(a_k^2), white noise of
   height E_k / d.  E_{>8} is computed from the trace of K^(>8) at d_decomp, which includes every
   degree up to the truncation.

Usage:  python -m experiments.fig2_activations.summarize --config paper
"""

import argparse
import json
import os

import numpy as np

NAMES = {"relu": "ReLU", "heaviside": "Heaviside", "tanh": "$$\\tanh$$", "bump": "$$y^2e^{-y^2}$$"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    R = np.load(os.path.join(args.results, f"fig2_activations_{args.config}.npz"))
    meta = json.loads(str(R["meta"]))
    ds, dd, dmax = meta["ds"], meta["d_decomp"], max(meta["ds"])
    js = [j for j in (1, 2, 10, 100) if j <= min(ds) // 2]
    for scale, sig2 in meta["scales"].items():
        print(f"\n**Scale $$\\sigma^2 = {sig2:.2f}$$.** Convergence of the head (median ratio to $$d = {dmax}$$):\n")
        print("| activation | d | " + " | ".join(f"j={j}" for j in js) + " |")
        print("|---" * (len(js) + 2) + "|")
        for a, name in NAMES.items():
            ref = np.median(R[f"{scale}__{a}__d{dmax}__K"], 0)
            for d in ds[:-1]:
                r = np.median(R[f"{scale}__{a}__d{d}__K"] / ref[None, :d], 0)
                print(f"| {name} | {d} | " + " | ".join(f"{r[j - 1]:.2f}" for j in js) + " |")
        print(f"\nFloor and Hermite energy (energies at $$d = {dd}$$, as fractions of $$\\mathbb E f^2$$):\n")
        print("| activation | " + " | ".join(f"$$d\\,\\lambda_{{d/2}}$$, d={d}" for d in ds)
              + " | $$\\mathbb E f^2$$ | $$E_{\\ge 2}$$ | k=0 | k=1 | k=2 | k=3..8 | k>8 |")
        print("|---" * (len(ds) + 8) + "|")
        for a, name in NAMES.items():
            floors = [d * np.median(R[f"{scale}__{a}__d{d}__K"][:, d // 2 - 1]) for d in ds]
            E = np.median(R[f"{scale}__{a}__d{dd}__energy"], 0)
            Egt = np.median(dd * R[f"{scale}__{a}__d{dd}__gt{meta['remainder']}"].mean(1))
            tot = E[:9].sum() + Egt
            fr = [E[0], E[1], E[2], E[3:9].sum(), Egt]
            print(f"| {name} | " + " | ".join(f"{x:.4f}" for x in floors) + f" | {tot:.4f} | {tot - E[0] - E[1]:.4f} | "
                  + " | ".join(f"{x / tot:.3f}" for x in fr) + " |")


if __name__ == "__main__":
    main()
