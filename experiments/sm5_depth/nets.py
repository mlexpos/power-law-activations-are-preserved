"""Networks for the depth experiment.  Each forward pass returns the list of features (flattened,
one (batch, dim) tensor per recorded layer) and the logits.

* RandomFeatureMLP: iterated random features with a controlled scale.  Layer l maps h_{l-1} to
  f(c_l W_l^T h_{l-1}) [then an optional per-sample norm], with W_l i.i.d. N(0, 1) (numpy seeds, so
  independent of the backend), no biases, and c_l = 1 / sqrt(mean_n |h_{l-1,n}|^2) fixed once from the
  data.  So every layer's pre-activations have mean square 1, as in the random-feature model of the
  paper with tr Cov = 1 (Experiments 3 and 4).
* DefaultMLP: the MLP of the submitted supplement: nn.Linear layers with PyTorch's default
  initialization (weights and biases uniform on +-1/sqrt(fan_in)), activation, optional norm.
* VGGCNN and ResNet20: the submitted architectures, unchanged.  Their convolutions have no biases
  and, at initialization in eval mode, BatchNorm is the identity, so at initialization they are
  positively homogeneous: the weight scale does not affect the spectra.
"""

import numpy as np
import torch
import torch.nn as nn


def he3(x):
    return x**3 - 3 * x


ACTS = {"relu": torch.relu, "tanh": torch.tanh, "he3": he3}


class RMSNorm(nn.Module):
    """x / rms(x) per sample, with a learnable scale (1 at initialization), as in the submitted code."""

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.scale * x / torch.sqrt((x**2).mean(-1, keepdim=True) + self.eps)


def make_norm(norm, dim):
    return {"none": None, "rmsnorm": RMSNorm(dim), "layernorm": nn.LayerNorm(dim)}[norm]


class RandomFeatureMLP:
    """Not an nn.Module: built layer by layer on the whole dataset (see run.py)."""

    def __init__(self, input_dim, width, depth, activation, norm, seed, tag=5):
        self.ws = [np.random.default_rng([tag, seed, l]).standard_normal((input_dim if l == 0 else width, width))
                   for l in range(depth)]
        self.act = ACTS[activation]
        self.norm = norm

    def features(self, X, device):
        """X: (n, input_dim) torch tensor (float64).  Returns [h_1, ..., h_L]."""
        out, h = [], X
        for W in self.ws:
            c = 1.0 / torch.sqrt((h**2).sum(1).mean())
            h = self.act(c * (h @ torch.as_tensor(W, dtype=h.dtype, device=device)))
            if self.norm == "rmsnorm":
                h = h / torch.sqrt((h**2).mean(1, keepdim=True))
            elif self.norm == "layernorm":
                h = h - h.mean(1, keepdim=True)
                h = h / torch.sqrt((h**2).mean(1, keepdim=True) + 1e-5)
            out.append(h)
        return out


class DefaultMLP(nn.Module):
    def __init__(self, input_dim=3072, width=1024, depth=4, activation="relu", norm="none", num_classes=10):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(input_dim if l == 0 else width, width) for l in range(depth)])
        self.norms = nn.ModuleList([make_norm(norm, width) or nn.Identity() for _ in range(depth)])
        self.act = ACTS[activation]
        self.head = nn.Linear(width, num_classes)

    def forward(self, x):
        h, feats = x.flatten(1), []
        for lin, nrm in zip(self.lins, self.norms):
            h = nrm(self.act(lin(h)))
            feats.append(h)
        return feats, self.head(h)


def _conv_bn_relu(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]


class VGGCNN(nn.Module):
    """Four blocks [conv-BN-ReLU, conv-BN-ReLU, max-pool], channels 64-128-256-512; features after each
    block (dimensions 16384, 8192, 4096, 2048)."""

    def __init__(self, num_classes=10):
        super().__init__()
        chans = [3, 64, 128, 256, 512]
        self.blocks = nn.ModuleList([nn.Sequential(*_conv_bn_relu(a, b), *_conv_bn_relu(b, b), nn.MaxPool2d(2))
                                     for a, b in zip(chans[:-1], chans[1:])])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        feats = []
        for b in self.blocks:
            x = b(x)
            feats.append(x.flatten(1))
        return feats, self.fc(x.mean((2, 3)))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet20(nn.Module):
    """CIFAR ResNet-20: conv1, then 3 groups of 3 basic blocks (16, 32, 64 channels); features after
    conv1 and after every block (dimensions 16384, 16384 x3, 8192 x3, 4096 x3)."""

    LAYERS = ["conv1"] + [f"G{g}-B{b}" for g in (1, 2, 3) for b in (1, 2, 3)]

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        blocks, cin = [], 16
        for cout, stride in [(16, 1), (32, 2), (64, 2)]:
            for b in range(3):
                blocks.append(BasicBlock(cin, cout, stride if b == 0 else 1))
                cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        feats = [x.flatten(1)]
        for b in self.blocks:
            x = b(x)
            feats.append(x.flatten(1))
        return feats, self.fc(x.mean((2, 3)))


LAYER_NAMES = {
    "mlp": [f"L{l}" for l in range(1, 5)],
    "cnn": [f"B{b}" for b in range(1, 5)],
    "resnet": ResNet20.LAYERS,
}
