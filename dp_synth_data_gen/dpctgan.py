from typing import Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

def _mlp(d_in, d_out, widths=(256,128), last_act=None):
    layers = [nn.Linear(d_in, widths[0]), nn.ReLU(),
              nn.Linear(widths[0], widths[1]), nn.ReLU(),
              nn.Linear(widths[1], d_out)]
    if last_act == "tanh":
        layers += [nn.Tanh()]
    return nn.Sequential(*layers)

class _Gen(nn.Module):
    def __init__(self, d_z, d_cond, d_out):
        super().__init__()
        self.net = _mlp(d_z + d_cond, d_out)

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))

class _Disc(nn.Module):
    def __init__(self, d_in, d_cond):
        super().__init__()
        self.net = _mlp(d_in + d_cond, 1)

    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=1))

class DPCTGAN:
    """
    Minimal DP-CTGAN-like trainer with Opacus.
    - One-hot+scaled features; conditional on label if provided.
    - Uses DP-SGD on *both* G and D (same σ and clip).
    """
    def __init__(self,
                 epochs=150,
                 batch_size=256,
                 max_grad_norm=1.0,
                 noise_multiplier=1.0,  # σ
                 delta=1e-5,
                 lr_g=2e-4,
                 lr_d=2e-4,
                 z_dim=64,
                 pac=16,
                 device: Optional[str]="auto"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.z_dim = z_dim
        self.pac = pac
        self.device = device
        self._columns: List[str] = []
        self._d_cond = 0
        self._fitted = False

    def _to_matrix(self, X: pd.DataFrame) -> np.ndarray:
        cats = [c for c in X.columns if X[c].dtype == "object"]
        nums = [c for c in X.columns if c not in cats]
        X_proc = pd.get_dummies(X, columns=cats, dummy_na=True)
        for c in nums:
            mu, sd = X_proc[c].mean(), X_proc[c].std() + 1e-6
            X_proc[c] = (X_proc[c] - mu) / sd
        self._columns = list(X_proc.columns)
        return X_proc.values.astype(np.float32)

    def _cond_vec(self, y: Optional[pd.Series], n_rows: int):
        if y is None:
            self._d_cond = 0
            return np.zeros((n_rows, 0), dtype=np.float32)
        # binary label -> one-hot (2). Generalize as needed.
        yy = y.values.astype(int)
        d = int(yy.max() + 1)
        onehot = np.eye(d, dtype=np.float32)[yy]
        self._d_cond = d
        return onehot

    def _from_matrix(self, M: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(M, columns=self._columns)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series]=None):
        X_mat = self._to_matrix(X)
        C_mat = self._cond_vec(y, X_mat.shape[0])
        device = torch.device("cuda" if (self.device == "auto" and torch.cuda.is_available()) else "cpu")

        # Enforce pac compatibility
        assert self.batch_size % self.pac == 0, "batch_size must be divisible by pac"

        d_in = X_mat.shape[1]
        d_cond = C_mat.shape[1]
        self._d_cond = d_cond

        G = _Gen(self.z_dim, d_cond, d_in).to(device)
        D = _Disc(d_in, d_cond).to(device)

        optG = torch.optim.Adam(G.parameters(), lr=self.lr_g, betas=(0.5, 0.9))
        optD = torch.optim.Adam(D.parameters(), lr=self.lr_d, betas=(0.5, 0.9))

        # Data loader
        X_tensor = torch.from_numpy(X_mat)
        C_tensor = torch.from_numpy(C_mat) if d_cond > 0 else torch.zeros((X_mat.shape[0], 0), dtype=torch.float32)
        ds = TensorDataset(X_tensor, C_tensor)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Make both optimizers private
        peD = PrivacyEngine()
        D, optD, dl = peD.make_private(
            module=D, optimizer=optD, data_loader=dl,
            noise_multiplier=self.noise_multiplier, max_grad_norm=self.max_grad_norm,
        )
        peG = PrivacyEngine()
        # For generator we create a synthetic loader with same batch size for DP-SGD bookkeeping
        dummy = DataLoader(TensorDataset(torch.zeros((self.batch_size, 1))), batch_size=self.batch_size)
        G, optG, _ = peG.make_private(
            module=G, optimizer=optG, data_loader=dummy,
            noise_multiplier=self.noise_multiplier, max_grad_norm=self.max_grad_norm,
        )

        bce = nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            for xb, cb in dl:
                xb = xb.to(device)
                cb = cb.to(device)

                # ---- Train Discriminator ----
                optD.zero_grad(set_to_none=True)
                z = torch.randn(xb.size(0), self.z_dim, device=device)
                x_fake = G(z, cb)
                y_real = torch.ones(xb.size(0), 1, device=device)
                y_fake = torch.zeros(xb.size(0), 1, device=device)
                d_real = D(xb, cb)
                d_fake = D(x_fake.detach(), cb)
                lossD = bce(d_real, y_real) + bce(d_fake, y_fake)
                lossD.backward()
                optD.step()

                # ---- Train Generator ----
                optG.zero_grad(set_to_none=True)
                z = torch.randn(xb.size(0), self.z_dim, device=device)
                x_fake = G(z, cb)
                d_fake = D(x_fake, cb)
                lossG = bce(d_fake, y_real)
                lossG.backward()
                optG.step()

        self._G = G
        self._device = device
        self._fitted = True
        return self

    def sample(self, n: int) -> pd.DataFrame:
        assert self._fitted, "Call fit() first."
        self._G.eval()
        with torch.no_grad():
            z = torch.randn(n, self.z_dim, device=self._device)
            # If conditional dimension > 0, default to class 1 distributionally (or sample prior)
            if self._d_cond > 0:
                # uniform over classes; adjust as needed
                c_idx = torch.randint(0, self._d_cond, (n,), device=self._device)
                C = torch.nn.functional.one_hot(c_idx, num_classes=self._d_cond).float()
            else:
                C = torch.zeros((n, 0), device=self._device)
            x_fake = self._G(z, C).cpu().numpy()
        return self._from_matrix(x_fake)
