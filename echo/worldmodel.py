"""Action-conditioned world model  s_{t+1} = f_theta(s_t, a_t)  (structured state, MLP ensemble).

Each member predicts the normalised state *delta*; the ensemble mean is the prediction and the
member disagreement is an uncertainty estimate.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from .wm_data import STATE_DIM, ACTION_DIM


class MLP(nn.Module):
    def __init__(self, hidden=256, layers=3, action_dim=ACTION_DIM):
        super().__init__()
        dims = [STATE_DIM + action_dim] + [hidden] * layers
        mods = []
        for i in range(layers):
            mods += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        mods.append(nn.Linear(hidden, STATE_DIM))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class WorldModel:
    def __init__(self, n_models: int = 5, hidden: int = 256, layers: int = 3, action_dim: int = ACTION_DIM):
        self.action_dim = action_dim
        self.models = [MLP(hidden, layers, action_dim) for _ in range(n_models)]
        self.x_mu = self.x_sd = self.d_mu = self.d_sd = None

    # -- normalisation statistics (fit on training data only)
    def fit_norm(self, S, A, S2):
        X = np.concatenate([S, A], 1)
        D = S2 - S
        self.x_mu, self.x_sd = X.mean(0), X.std(0) + 1e-6
        self.d_mu, self.d_sd = D.mean(0), D.std(0) + 1e-6

    def _x(self, S, A):
        X = np.concatenate([S, A], 1)
        return torch.as_tensor((X - self.x_mu) / self.x_sd, dtype=torch.float32)

    def train(self, S, A, S2, epochs=200, lr=1e-3, bs=128, seed=0, verbose=True, val=None):
        X = self._x(S, A)
        Y = torch.as_tensor(((S2 - S) - self.d_mu) / self.d_sd, dtype=torch.float32)
        for k, m in enumerate(self.models):
            g = torch.Generator().manual_seed(seed + k)
            opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
            for ep in range(epochs):
                perm = torch.randperm(len(X), generator=g)   # each member: own shuffling, own init
                m.train()
                for i in range(0, len(X), bs):
                    idx = perm[i:i + bs]
                    loss = nn.functional.mse_loss(m(X[idx]), Y[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
                if verbose and (ep + 1) % 50 == 0:
                    print(f"  member {k} epoch {ep+1}/{epochs} train loss {loss.item():.4f}", flush=True)

    @torch.no_grad()
    def predict(self, S, A):
        """Returns (mean next state, per-dim std across members)."""
        X = self._x(S, A)
        outs = []
        for m in self.models:
            m.eval()
            outs.append(m(X).numpy() * self.d_sd + self.d_mu + S)
        outs = np.stack(outs)
        return outs.mean(0), outs.std(0)

    def save(self, path):
        torch.save({"states": [m.state_dict() for m in self.models],
                    "norm": (self.x_mu, self.x_sd, self.d_mu, self.d_sd),
                    "n": len(self.models), "action_dim": self.action_dim, "hidden": self.models[0].net[0].out_features,
                    "layers": sum(isinstance(l, nn.Linear) for l in self.models[0].net) - 1}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, weights_only=False)
        wm = cls(ck["n"], ck["hidden"], ck["layers"], ck.get("action_dim", ACTION_DIM))
        for m, sd in zip(wm.models, ck["states"]):
            m.load_state_dict(sd)
        wm.x_mu, wm.x_sd, wm.d_mu, wm.d_sd = ck["norm"]
        return wm
