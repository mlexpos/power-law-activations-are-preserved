"""Monte Carlo estimates of random-feature second-moment matrices, for inputs or activations with
no closed form.

    K_W ~ (1/d) (1/m) sum_n f(W^T x_n) f(W^T x_n)^T,   and the mean (1/m) sum_n f(W^T x_n).

The inputs x_n are always generated with numpy (seeded by the caller), so the estimate does not
depend on the backend; the matrix products run in float64 with numpy or torch.
"""

import numpy as np


class Accumulator:
    """Accumulates sum_n f(y_n) f(y_n)^T and sum_n f(y_n) for several activations on a common
    stream of pre-activations y_n = W^T x_n."""

    def __init__(self, W, activations, device="numpy"):
        self.acts = list(activations)
        self.device = device
        self.d = W.shape[1]
        self.m = 0
        if device == "numpy":
            self.W = W
            self.S = [np.zeros((self.d, self.d)) for _ in self.acts]
            self.s = [np.zeros(self.d) for _ in self.acts]
        else:
            import torch
            self.torch = torch
            self.W = torch.as_tensor(W, dtype=torch.float64, device=device)
            self.S = [torch.zeros(self.d, self.d, dtype=torch.float64, device=device) for _ in self.acts]
            self.s = [torch.zeros(self.d, dtype=torch.float64, device=device) for _ in self.acts]

    def add(self, X):
        """X: (n, v) numpy array of inputs."""
        if self.device != "numpy":
            X = self.torch.as_tensor(X, dtype=self.torch.float64, device=self.device)
        Y0 = X @ self.W
        for i, f in enumerate(self.acts):
            Y = f(Y0)
            self.S[i] += Y.T @ Y
            self.s[i] += Y.sum(0)
        self.m += X.shape[0]

    def result(self):
        """[(K, Kc)] per activation: second moment and covariance, both divided by d, as numpy."""
        out = []
        for S, s in zip(self.S, self.s):
            if self.device != "numpy":
                S, s = S.cpu().numpy(), s.cpu().numpy()
            K = S / self.m
            mu = s / self.m
            out.append((K / self.d, (K - np.outer(mu, mu)) / self.d))
        return out
