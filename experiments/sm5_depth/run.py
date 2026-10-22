"""Supplement SM5, redone: covariance spectra of CIFAR-10 features across the layers of MLPs, a VGG-style
CNN and ResNet-20, at random initialization and after training.

Parts (run separately or together):
  mlp      random-initialization MLPs (depth 4), five variants (ReLU, tanh, He3, ReLU+RMSNorm,
           ReLU+LayerNorm), in two setups:
             C  iterated random features with a controlled scale (nets.RandomFeatureMLP): centered
                CIFAR-10 scaled so that the mean of |x|^2 is 1, Gaussian weights, no biases, every
                layer rescaled so its pre-activations have mean square 1;
             D  the submitted setup (nets.DefaultMLP): inputs in [-1, 1], PyTorch default init with
                biases.  Also records the pre-activation scale per layer.
  conv     VGG-style CNN and ResNet-20 at initialization.
  trained  ReLU MLP (setup D), CNN and ResNet-20 trained on the 50,000 training images: SGD, lr 0.1,
           momentum 0.9, weight decay 1e-4, cosine schedule, batch 256, no augmentation, shuffled with
           a seeded generator.  Records the training history and the test accuracy.

For every network and layer: the eigenvalues of the centered covariance of the features over the
first n_eval training images (eval mode).  When the feature dimension exceeds n_eval (quick config
only), the nonzero eigenvalues are taken from the centered Gram matrix instead.

Usage (from the repository root):
    python -m experiments.sm5_depth.run --config quick [--part mlp|conv|trained|all]
    python -m experiments.sm5_depth.run --config paper --part all      # GPU
Output: results/sm5_depth_<part>_<config>.npz
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from experiments.sm5_depth.nets import LAYER_NAMES, DefaultMLP, RandomFeatureMLP, ResNet20, VGGCNN
from plrf.inputs import load_cifar10

CONFIGS = {
    "quick": dict(n_eval=2000, n_train=5000, width=256, depth=4, random_seeds=[0, 1], trained_seeds=[0],
                  epochs=1, batch=256, eval_batch=500),
    "paper": dict(n_eval=50000, n_train=50000, width=1024, depth=4, random_seeds=[0, 1, 2, 3, 4],
                  trained_seeds=[0, 1, 2], epochs=50, batch=256, eval_batch=500),
}
MLP_VARIANTS = {"relu": ("relu", "none"), "tanh": ("tanh", "none"), "he3": ("he3", "none"),
                "relu_rmsnorm": ("relu", "rmsnorm"), "relu_layernorm": ("relu", "layernorm")}


class Spectra:
    """Centered covariance spectra of several feature streams, accumulated over batches in float64."""

    def __init__(self, n_total, device):
        self.n_total, self.device = n_total, device
        self.S, self.s, self.F = {}, {}, {}
        self.n = 0

    def add(self, feats):
        for i, F in enumerate(feats):
            F = F.to(torch.float64)
            if F.shape[1] > self.n_total:          # Gram route (quick config only)
                self.F.setdefault(i, []).append(F.cpu())
            else:
                if i not in self.S:
                    self.S[i] = torch.zeros(F.shape[1], F.shape[1], dtype=torch.float64, device=self.device)
                    self.s[i] = torch.zeros(F.shape[1], dtype=torch.float64, device=self.device)
                self.S[i] += F.T @ F
                self.s[i] += F.sum(0)
        self.n += feats[0].shape[0]

    def eigenvalues(self):
        out = []
        for i in sorted(set(self.S) | set(self.F)):
            if i in self.S:
                mu = self.s[i] / self.n
                C = self.S[i] / self.n - torch.outer(mu, mu)
                lam = torch.linalg.eigvalsh(C).flip(0)
                del C
            else:
                F = torch.cat(self.F[i]).to(self.device)
                F = F - F.mean(0)
                lam = torch.linalg.eigvalsh(F @ F.T / self.n).flip(0)
            out.append(lam.cpu().numpy())
        return out


def input_spectrum(X, device):
    sp = Spectra(X.shape[0], device)
    sp.add([X])
    return sp.eigenvalues()[0]


def eval_features(model, X, cfg, device, image=True):
    """Spectra of every recorded layer of an nn.Module over X (n, 3072) in eval mode."""
    model.eval()
    sp = Spectra(X.shape[0], device)
    with torch.no_grad():
        for i in range(0, X.shape[0], cfg["eval_batch"]):
            xb = X[i:i + cfg["eval_batch"]].to(device)
            feats, _ = model(xb.view(-1, 3, 32, 32) if image else xb)
            sp.add(feats)
    return sp.eigenvalues()


def train(model, Xtr, ytr, Xte, yte, cfg, seed, device, image=True):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    lossf = torch.nn.CrossEntropyLoss()
    shape = (-1, 3, 32, 32) if image else (-1, 3072)
    hist = []
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(Xtr.shape[0], generator=g)
        tot, correct, ls = 0, 0, 0.0
        for i in range(0, len(perm), cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            xb, yb = Xtr[idx].to(device).view(shape), ytr[idx].to(device)
            opt.zero_grad()
            _, logits = model(xb)
            loss = lossf(logits, yb)
            loss.backward()
            opt.step()
            ls += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            tot += len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            te = sum((model(Xte[i:i + 1000].to(device).view(shape))[1].argmax(1) == yte[i:i + 1000].to(device)).sum().item()
                     for i in range(0, Xte.shape[0], 1000))
        hist.append(dict(epoch=epoch + 1, train_loss=ls / tot, train_acc=correct / tot, test_acc=te / Xte.shape[0]))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1}: train loss {ls / tot:.3f}, train acc {correct / tot:.3f}, test acc {te / Xte.shape[0]:.3f}",
                  flush=True)
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick", choices=CONFIGS)
    ap.add_argument("--part", default="all", choices=["mlp", "conv", "trained", "all"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    cfg = CONFIGS[args.config]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"config={args.config} {cfg} device={device}", flush=True)

    raw = load_cifar10(args.data)                       # (60000, 3072) in [0, 1]; train = first 50,000
    labels = _labels(args.data)
    Xpm = torch.tensor(raw * 2 - 1, dtype=torch.float32)  # the submitted preprocessing, [-1, 1]
    y = torch.tensor(labels, dtype=torch.long)
    Xtr, ytr = Xpm[:cfg["n_train"]], y[:cfg["n_train"]]
    Xte, yte = Xpm[50000:], y[50000:]
    Xev = Xpm[:cfg["n_eval"]]
    train_raw = raw[:50000]
    Xc = train_raw - train_raw.mean(0)
    Xc /= np.sqrt((Xc**2).sum(1).mean())                # centered, mean |x|^2 = 1 over the training set
    Xc_ev = torch.tensor(Xc[:cfg["n_eval"]], dtype=torch.float64, device=device)

    parts = ["mlp", "conv", "trained"] if args.part == "all" else [args.part]
    for part in parts:
        store, meta = {}, dict(cfg, config=args.config, part=part, device=device, layer_names=LAYER_NAMES)
        t0 = time.time()

        def put(group, layers, lams):
            for name, lam in zip(layers, lams):
                store.setdefault(f"{group}__{name}", []).append(lam)

        store["input__centered"] = [input_spectrum(Xc_ev, device)]
        if part == "mlp":
            for v, (act, norm) in MLP_VARIANTS.items():
                for seed in cfg["random_seeds"]:
                    net = RandomFeatureMLP(3072, cfg["width"], cfg["depth"], act, norm, seed)
                    feats = net.features(Xc_ev, device)
                    put(f"mlpC_{v}", LAYER_NAMES["mlp"], [_cov_eigs(F) for F in feats])
                    del feats
                    torch.manual_seed(seed)
                    m = DefaultMLP(3072, cfg["width"], cfg["depth"], act, norm).to(device)
                    put(f"mlpD_{v}", LAYER_NAMES["mlp"], eval_features(m, Xev, cfg, device, image=False))
                    store.setdefault(f"mlpD_{v}__preact_rms", []).append(_preact_rms(m, Xev[:5000].to(device)))
                print(f"  mlp {v}: done ({time.time() - t0:.0f}s)", flush=True)
        elif part == "conv":
            for name, cls in [("cnn", VGGCNN), ("resnet", ResNet20)]:
                for seed in cfg["random_seeds"]:
                    torch.manual_seed(seed)
                    put(f"{name}_random", LAYER_NAMES[name], eval_features(cls().to(device), Xev, cfg, device))
                    print(f"  {name} seed {seed}: done ({time.time() - t0:.0f}s)", flush=True)
        else:
            hists = {}
            for name, make, image in [("mlp", lambda: DefaultMLP(3072, cfg["width"], cfg["depth"]), False),
                                      ("cnn", VGGCNN, True), ("resnet", ResNet20, True)]:
                for seed in cfg["trained_seeds"]:
                    print(f"  training {name}, seed {seed}", flush=True)
                    torch.manual_seed(seed)
                    model = make().to(device)
                    hists[f"{name}_seed{seed}"] = train(model, Xtr, ytr, Xte, yte, cfg, seed, device, image)
                    put(f"{name}_trained", LAYER_NAMES[name], eval_features(model, Xev, cfg, device, image))
                    print(f"  {name} seed {seed}: test acc {hists[f'{name}_seed{seed}'][-1]['test_acc']:.3f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
            meta["training_history"] = hists

        os.makedirs(args.outdir, exist_ok=True)
        out = os.path.join(args.outdir, f"sm5_depth_{part}_{args.config}.npz")
        meta["seed_rule"] = ("RandomFeatureMLP: W_l ~ numpy.random.default_rng([5, seed, l]); all torch models: "
                             "torch.manual_seed(seed) before construction; training order: "
                             "torch.Generator().manual_seed(seed)")
        np.savez_compressed(out, meta=json.dumps(meta), **{k: np.array(v) for k, v in store.items()})
        print(f"saved {out}", flush=True)


def _labels(data_dir):
    import pickle
    import warnings
    root = os.path.join(data_dir, "cifar-10-batches-py")
    ys = []
    for b in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
        with open(os.path.join(root, b), "rb") as f, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ys.append(np.array(pickle.load(f, encoding="bytes")[b"labels"]))
    return np.concatenate(ys)


def _cov_eigs(F):
    F = F - F.mean(0)
    return torch.linalg.eigvalsh(F.T @ F / F.shape[0]).flip(0).cpu().numpy()


def _preact_rms(model, X):
    """Root mean square of the pre-activations of each layer of a DefaultMLP."""
    out, h = [], X
    with torch.no_grad():
        for lin, nrm in zip(model.lins, model.norms):
            z = lin(h)
            out.append(float(z.pow(2).mean().sqrt()))
            h = nrm(model.act(z))
    return np.array(out)


if __name__ == "__main__":
    main()
