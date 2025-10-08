from typing import Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

class _VAE(nn.Module):
    def __init__(self, d_in, d_latent=16, widths=(256,128)):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, widths[0]), nn.ReLU(),
            nn.Linear(widths[0], widths[1]), nn.ReLU(),
        )
        self.mu = nn.Linear(widths[1], d_latent)
        self.logvar = nn.Linear(widths[1], d_latent)
        self.dec = nn.Sequential(
            nn.Linear(d_latent, widths[1]), nn.ReLU(),
            nn.Linear(widths[1], widths[0]), nn.ReLU(),
            nn.Linear(widths[0], d_in),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar

def _vae_loss(x, x_hat, mu, logvar):
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld

class DPTVAE:
    """
    Differentially-Private TVAE-like baseline (plain VAE with DP-SGD).
    - Trains on numerical matrix (one-hot cats + scaled nums).
    - Uses Opacus PrivacyEngine to clip per-sample grads and add Gaussian noise.
    """
    def __init__(self,
                 epochs=150,
                 batch_size=256,
                 max_grad_norm=1.0,
                 noise_multiplier=1.0,     # σ
                 delta=1e-5,
                 lr=1e-3,
                 latent_dim=16,
                 device: Optional[str]="auto"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.lr = lr
        self.latent_dim = latent_dim
        self.device = device
        self._columns: List[str] = []
        self._model = None
        self._fitted = False

    def _to_matrix(self, X: pd.DataFrame) -> np.ndarray:
        # Basic: one-hot for object cols, z-score for numeric
        cats = [c for c in X.columns if X[c].dtype == "object"]
        nums = [c for c in X.columns if c not in cats]
        X_proc = pd.get_dummies(X, columns=cats, dummy_na=True)
        # z-score numeric original cols only (not one-hots)
        for c in nums:
            mu, sd = X_proc[c].mean(), X_proc[c].std() + 1e-6
            X_proc[c] = (X_proc[c] - mu) / sd
        self._columns = list(X_proc.columns)
        return X_proc.values.astype(np.float32)

    def _from_matrix(self, M: np.ndarray) -> pd.DataFrame:
        # Return numeric matrix as DF; (decoding cats requires metadata; keep simple)
        return pd.DataFrame(M, columns=self._columns)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series]=None):
        X_mat = self._to_matrix(X)
        device = torch.device("cuda" if (self.device == "auto" and torch.cuda.is_available()) else "cpu")
        d_in = X_mat.shape[1]
        model = _VAE(d_in=d_in, d_latent=self.latent_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        ds = TensorDataset(torch.from_numpy(X_mat))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # Attach PrivacyEngine
        pe = PrivacyEngine()
        model, opt, dl = pe.make_private(
            module=model,
            optimizer=opt,
            data_loader=dl,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        model.train()
        for _ in range(self.epochs):
            for (xb,) in dl:
                xb = xb.to(device)
                opt.zero_grad(set_to_none=True)
                x_hat, mu, logvar = model(xb)
                loss = _vae_loss(xb, x_hat, mu, logvar)
                loss.backward()
                opt.step()

        self._model = model
        self._device = device
        self._fitted = True
        return self

    def sample(self, n: int) -> pd.DataFrame:
        assert self._fitted, "Call fit() first."
        self._model.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=self._device)
            x_hat = self._model.dec(z).cpu().numpy()
        return self._from_matrix(x_hat)
