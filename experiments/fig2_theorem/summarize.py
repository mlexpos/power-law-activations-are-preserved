"""Print the summary table quoted on the documentation page.

For each activation and d: the median over sketches of lambda_j(K_W) / lambda_j(population) at a
few indices j, and J_*(d), the largest J such that the median ratio lies in [0.8, 1.25] for all
j <= J.

Usage:  python -m experiments.fig2_theorem.summarize --config paper
"""

import argparse
import json
import os

import numpy as np

NAMES = {"y1": "$$y$$", "y2": "$$y^2$$", "y2c": "$$y^2$$ centered", "y3": "$$y^3$$",
         "cub": "$$y + y^2/2 + y^3/6$$", "he3": "$$\\mathrm{He}_3(y) = y^3 - 3y$$"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick")
    ap.add_argument("--results", default="results")
    ap.add_argument("--band", type=float, default=1.25)
    args = ap.parse_args()
    R = np.load(os.path.join(args.results, f"fig2_theorem_{args.config}.npz"))
    ds = json.loads(str(R["meta"]))["ds"]
    js = [1, 10, 100, 1000]
    print(f"| activation | d | " + " | ".join(f"ratio at j={j}" for j in js) + f" | J_* (ratio in [{1/args.band:.2f}, {args.band}]) |")
    print("|---" * (len(js) + 3) + "|")
    for key, name in NAMES.items():
        for d in ds:
            ratio = np.median(R[f"lam_{key}_d{d}"] / R[f"pop_{key}_d{d}"][None, :], axis=0)
            ok = (ratio >= 1 / args.band) & (ratio <= args.band)
            J = int(np.argmin(ok)) if not ok.all() else d
            cells = [f"{ratio[j - 1]:.2f}" if j <= d else "–" for j in js]
            print(f"| {name} | {d} | " + " | ".join(cells) + f" | {J} |")


if __name__ == "__main__":
    main()
