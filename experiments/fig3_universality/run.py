"""Figure 3, redone: does the random-feature spectrum depend on the input distribution beyond its
covariance?

Two comparisons, each over n_draws independent sketches W, with the same W for every
distribution within a draw (so ratios between distributions are paired):

* synthetic: x = H^{1/2} xi, H = diag(j^{-alpha}) / tr, v = 2d, xi_k i.i.d. Gaussian, Rademacher,
  Student-t(9) or Student-t(3), standardized.  No per-sample normalization.
* CIFAR-10: all 60,000 images, centered per pixel and scaled so that the mean of |x|^2 is 1
  (v = 3072), against a Gaussian with the same covariance Sigma.

Activations: y^2, y^3 (polynomial), ReLU, tanh, y^2 exp(-y^2).  The identity is omitted: for it
K_W = W^T Cov(x) W / d is the same for every distribution with the same covariance.

How each matrix K_W = (1/d) E_x[f(W^T x) f(W^T x)^T] is computed:
* polynomial f, synthetic inputs: exactly, for every distribution (plrf.exact for Gaussian,
  plrf.iid otherwise).  Student-t(3) is not used: K_W needs E xi^{2p}, so nu > 2p.
* ReLU, Gaussian inputs (synthetic and CIFAR surrogate): exactly, by the arc-cosine kernel.
* everything else: Monte Carlo with m samples.  For a fair comparison the Gaussian case is also
  estimated by Monte Carlo with the same m; ratios are taken Monte Carlo to Monte Carlo.
  Student-t(3) only with the bounded activations tanh and y^2 exp(-y^2).
* CIFAR: the average over the 60,000 images (exact for the empirical distribution), and the
  Gaussian surrogate by Monte Carlo with m = 60,000 (matched sample size) and exactly where possible.
Both the second moment K_W and the covariance (K_W minus the rank-one mean term) are stored.

Usage (from the repository root):
    python -m experiments.fig3_universality.run --config quick
    python -m experiments.fig3_universality.run --config paper      # GPU recommended
Output: results/fig3_universality_<config>.npz
"""

import argparse
import json
import os
import time

import numpy as np

from plrf.activations import BUMP, RELU, TANH, monomial
from plrf.exact import mean_vector, relu_second_moment, second_moment
from plrf.iid import second_moment_and_mean_iid
from plrf.inputs import GAUSSIAN, RADEMACHER, load_cifar10, preprocess_cifar, student_t, unit_trace_power_law
from plrf.montecarlo import Accumulator

CONFIGS = {
    "quick": dict(alpha=1.31, d=500, v_over_d=2, m_over_d=20, n_draws=3, chunk=5000),
    "paper": dict(alpha=1.31, d=3000, v_over_d=2, m_over_d=20, n_draws=20, chunk=5000),
}
ACTS = {"y2": monomial(2), "y3": monomial(3), "relu": RELU, "tanh": TANH, "bump": BUMP}
POLY = ["y2", "y3"]
NONPOLY = ["relu", "tanh", "bump"]
BOUNDED = ["tanh", "bump"]
COORDS = {"gauss": GAUSSIAN, "rad": RADEMACHER, "t9": student_t(9), "t3": student_t(3)}
# integer codes used in the seeds
CODE = {"gauss": 0, "rad": 1, "t9": 2, "t3": 3, "gsur": 4}
TAG = 3  # Figure 3


def w_seed(which, d, draw):
    """W (synthetic: which = 0; CIFAR: which = 1) ~ numpy.random.default_rng([3, which, d, draw])."""
    return [TAG, which, d, draw]


def x_seed(dist, d, draw):
    """Monte Carlo inputs for distribution `dist` ~ numpy.random.default_rng([3, 2, CODE[dist], d, draw])."""
    return [TAG, 2, CODE[dist], d, draw]


class Eig:
    def __init__(self, device):
        self.device = device

    def __call__(self, K):
        if self.device == "numpy":
            return np.linalg.eigvalsh(K)[::-1].copy()
        import torch
        Kt = torch.as_tensor(K, dtype=torch.float64, device=self.device)
        return torch.linalg.eigvalsh(Kt).flip(0).cpu().numpy()


def chunks(m, chunk):
    return [min(chunk, m - s) for s in range(0, m, chunk)]


def mc_spectra(W, batches, acts, device, eig):
    """Second moments of the given activations averaged over the input batches (each (n, v));
    returns {act: (lam_K, lam_C)}."""
    acc = Accumulator(W, [ACTS[a] for a in acts], device)
    for X in batches:
        acc.add(X)
    out = {a: (eig(K), eig(C)) for a, (K, C) in zip(acts, acc.result())}
    del acc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick", choices=CONFIGS)
    ap.add_argument("--device", default="auto", help="auto | numpy | cpu | cuda")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--data", default="data", help="directory holding (or to download) cifar-10-batches-py")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "numpy"
        except ImportError:
            device = "numpy"
    print(f"config={args.config} {cfg} backend={device}", flush=True)
    eig = Eig(device)

    d, v, m, chunk = cfg["d"], cfg["v_over_d"] * cfg["d"], cfg["m_over_d"] * cfg["d"], cfg["chunk"]
    H = unit_trace_power_law(v, cfg["alpha"])
    Xc, Sigma = preprocess_cifar(load_cifar10(args.data))
    N, vc = Xc.shape
    h, U = np.linalg.eigh(Sigma)
    h = np.clip(h, 0, None)
    Sigma_half = (U * np.sqrt(h)) @ U.T
    store = {"H": H, "cifar_spectrum": h[::-1].copy()}
    lam = {}

    def put(key, pair):
        for kind, x in zip("KC", pair):
            lam.setdefault(f"{key}__{kind}", []).append(x)

    for draw in range(cfg["n_draws"]):
        t0 = time.time()
        # ---- synthetic inputs ----
        W = np.random.default_rng(w_seed(0, d, draw)).standard_normal((v, d))
        G = W.T @ (H[:, None] * W)
        for a in POLY:
            f = ACTS[a]
            K = second_moment(G, f)
            put(f"syn__{a}__gauss__exact", (eig(K), eig(K - np.outer(*[mean_vector(G, f)] * 2) / d)))
            for dist in ["rad", "t9"]:
                K, mu = second_moment_and_mean_iid(W, H, f, COORDS[dist])
                put(f"syn__{a}__{dist}__exact", (eig(K), eig(K - np.outer(mu, mu) / d)))
        put("syn__relu__gauss__exact", (eig(relu_second_moment(G)), eig(relu_second_moment(G, centered=True))))
        del G
        sqH = np.sqrt(H)
        for dist in ["gauss", "rad", "t9", "t3"]:
            acts = BOUNDED if dist == "t3" else NONPOLY
            rng = np.random.default_rng(x_seed(dist, d, draw))
            batches = (COORDS[dist].sample(rng, (n, v)) * sqH for n in chunks(m, chunk))
            res = mc_spectra(W, batches, acts, device, eig)
            for a, pair in res.items():
                put(f"syn__{a}__{dist}__mc", pair)
        t1 = time.time()

        # ---- CIFAR-10 and its Gaussian surrogate ----
        Wc = np.random.default_rng(w_seed(1, d, draw)).standard_normal((vc, d))
        res = mc_spectra(Wc, (Xc[s:s + chunk] for s in range(0, N, chunk)), list(ACTS), device, eig)
        for a, pair in res.items():
            put(f"cifar__{a}__cifar__data", pair)
        Gc = Wc.T @ Sigma @ Wc
        for a in POLY:
            f = ACTS[a]
            K = second_moment(Gc, f)
            put(f"cifar__{a}__gsur__exact", (eig(K), eig(K - np.outer(*[mean_vector(Gc, f)] * 2) / d)))
        put("cifar__relu__gsur__exact", (eig(relu_second_moment(Gc)), eig(relu_second_moment(Gc, centered=True))))
        del Gc
        rng = np.random.default_rng(x_seed("gsur", d, draw))
        batches = (rng.standard_normal((n, vc)) @ Sigma_half for n in chunks(N, chunk))
        res = mc_spectra(Wc, batches, list(ACTS), device, eig)
        for a, pair in res.items():
            put(f"cifar__{a}__gsur__mc", pair)
        print(f"draw {draw + 1}/{cfg['n_draws']}: synthetic {t1 - t0:.0f}s, CIFAR {time.time() - t1:.0f}s", flush=True)

    store.update({k: np.array(x) for k, x in lam.items()})
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"fig3_universality_{args.config}.npz")
    meta = dict(cfg, config=args.config, backend=device, v=v, m=m, cifar_N=N, cifar_v=vc,
                activations={k: f.name for k, f in ACTS.items()},
                distributions={k: c.name for k, c in COORDS.items()},
                key_format="<set>__<activation>__<distribution>__<method>__<K|C>; K = second moment, "
                           "C = covariance; arrays are (n_draws, d), eigenvalues in decreasing order",
                seed_rule="W ~ default_rng([3, 0, d, draw]) (synthetic, (v, d)) and default_rng([3, 1, d, draw]) "
                          "(CIFAR, (3072, d)); Monte Carlo inputs ~ default_rng([3, 2, code, d, draw]) with "
                          f"code {CODE}, drawn in chunks of {chunk} rows")
    np.savez_compressed(out, meta=json.dumps(meta), **store)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
