"""Print the summary table quoted on the documentation page.

For each activation and each non-Gaussian input: the median over sketches of the paired ratio
lambda_j(K_W; distribution) / lambda_j(K_W; Gaussian) at a few indices j, and J^0, the smallest J
such that the median ratio lies in [0.8, 1.25] for every J <= j <= d/2 (the edge j > d/2 is left
out).  So J^0 = 1 means agreement within 25% over the whole range, and a large J^0 means the head of
the spectrum depends on the distribution.

Usage:  python -m experiments.fig3_universality.summarize --config paper [--kind C]
"""

import argparse
import json
import os

import numpy as np

ACT = {"y2": "$$y^2$$", "y3": "$$y^3$$", "relu": "ReLU", "tanh": "$$\\tanh$$", "bump": "$$y^2e^{-y^2}$$"}
DIST = {"rad": "Rademacher", "t9": "Student-$$t_9$$", "t3": "Student-$$t_3$$", "cifar": "CIFAR-10"}
# (numerator, denominator) keys without the __K / __C suffix
ROWS = (
    [(f"syn__{a}__{x}__exact", f"syn__{a}__gauss__exact") for a in ("y2", "y3") for x in ("rad", "t9")]
    + [(f"syn__{a}__{x}__mc", f"syn__{a}__gauss__mc") for a in ("relu", "tanh", "bump") for x in ("rad", "t9", "t3")]
    + [(f"cifar__{a}__cifar__data", f"cifar__{a}__gsur__mc") for a in ACT]
)
CHECKS = [(f"syn__relu__gauss__mc", "syn__relu__gauss__exact")] + \
         [(f"cifar__{a}__gsur__mc", f"cifar__{a}__gsur__exact") for a in ("y2", "y3", "relu")]


def j0(ratio, band, jmax):
    ok = (ratio[:jmax] >= 1 / band) & (ratio[:jmax] <= band)
    bad = np.nonzero(~ok)[0]
    return 1 if len(bad) == 0 else int(bad[-1]) + 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick")
    ap.add_argument("--results", default="results")
    ap.add_argument("--kind", default="K", choices="KC")
    ap.add_argument("--band", type=float, default=1.25)
    args = ap.parse_args()
    R = np.load(os.path.join(args.results, f"fig3_universality_{args.config}.npz"))
    d = json.loads(str(R["meta"]))["d"]
    js = [j for j in (1, 10, 100, 1000) if j <= d // 2]

    def ratio(num, den):
        return np.median(R[f"{num}__{args.kind}"] / R[f"{den}__{args.kind}"], axis=0)

    print(f"| activation | input | " + " | ".join(f"ratio at j={j}" for j in js)
          + f" | J^0 (ratio in [{1/args.band:.2f}, {args.band}] for J^0 <= j <= d/2) |")
    print("|---" * (len(js) + 3) + "|")
    for num, den in [r for r in ROWS if f"{r[0]}__{args.kind}" in R.files]:
        _, a, x, _ = num.split("__")
        r = ratio(num, den)
        print(f"| {ACT[a]} | {DIST[x]} | " + " | ".join(f"{r[j - 1]:.2f}" for j in js) + f" | {j0(r, args.band, d // 2)} |")

    print("\nMonte Carlo against closed form (Gaussian inputs): max |ratio - 1| of the medians over j <= d/2")
    for num, den in CHECKS:
        r = ratio(num, den)[: d // 2]
        print(f"  {num:28s} {np.abs(r - 1).max():.3f}")


if __name__ == "__main__":
    main()
