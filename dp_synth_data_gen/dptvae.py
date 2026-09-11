# dp_synth_data_gen/dptvae.py
# Conditional VAE with DP-SGD (Opacus) for tabular synthesis.
#
# Rationale: GAN-based DP synthesis (DPCTGAN) trains an adversarial minimax
# game where one side's gradients are clipped+noised by DP-SGD; that noise
# destabilizes the game far more than it would a single smooth objective.
# A VAE has no adversary -- encoder+decoder are one network trained with one
# loss, so DP-SGD noise degrades it more gracefully. This also fixes two
# real bugs in the previous stub: it never modeled the label at all, and
# sample() returned a raw one-hot/z-scored matrix with no way to invert it
# back to the original schema (would crash run_tuner's downstream eval).
#
# Feature encoding: numeric columns with many unique values get a
# QuantileTransformer -> N(0,1) (MSE reconstruction loss). Columns that are
# either non-numeric or numeric with low cardinality (binary flags, ordinal
# buckets like GenHlth/Education/Age-group) are treated as categorical:
# one-hot targets, reconstructed via per-column cross-entropy. Most columns
# in this dataset are binary/low-cardinality, so this matters a lot more
# than it would on a mostly-continuous dataset.

from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.preprocessing import QuantileTransformer

CAT_CARDINALITY_THRESHOLD = 15  # numeric cols with <= this many unique values -> categorical


class _CVAE(nn.Module):
    def __init__(self, d_in: int, d_cond: int, d_latent: int, widths: Tuple[int, int] = (256, 128)):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in + d_cond, widths[0]), nn.ReLU(),
            nn.Linear(widths[0], widths[1]), nn.ReLU(),
        )
        self.mu = nn.Linear(widths[1], d_latent)
        self.logvar = nn.Linear(widths[1], d_latent)
        self.dec = nn.Sequential(
            nn.Linear(d_latent + d_cond, widths[1]), nn.ReLU(),
            nn.Linear(widths[1], widths[0]), nn.ReLU(),
            nn.Linear(widths[0], d_in),
        )

    def encode(self, x, c):
        h = self.enc(torch.cat([x, c], dim=1))
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.dec(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar


class DPTVAE:
    """
    Differentially-private conditional VAE for tabular data.
    Public API mirrors DPCTGAN (dpctgan_v2.py): fit(X, y), sample(n, return_y, y_cond), sample_labels(n).
    """

    def __init__(
        self,
        epochs: int = 150,
        batch_size: int = 256,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        delta: float = 1e-5,
        lr: float = 1e-3,
        latent_dim: int = 16,
        beta: float = 0.02,          # KL weight; low so reconstruction fidelity dominates under DP noise
        device: Optional[str] = "auto",
        secure_mode: bool = False,
        _num_quantiles: int = 1000,
        **kwargs,
    ):
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.delta = float(delta)
        self.lr = float(lr)
        self.latent_dim = int(latent_dim)
        self.beta = float(beta)
        self.device = device
        self.secure_mode = secure_mode
        self._num_quantiles = int(_num_quantiles)

        self._orig_columns: List[str] = []
        self._num_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._cat_values: Dict[str, List[str]] = {}
        self._feature_slices: Dict[str, slice] = {}
        self._num_tx: Dict[str, QuantileTransformer] = {}
        self._num_min: Dict[str, float] = {}
        self._num_max: Dict[str, float] = {}

        self._model: Optional[_CVAE] = None
        self._device: Optional[torch.device] = None
        self._d_cond: int = 0
        self._fitted: bool = False

        self._has_y: bool = False
        self._y_classes: Optional[np.ndarray] = None
        self._y_probs: Optional[np.ndarray] = None

    # ---- schema ----
    def _remember_columns(self, X):
        self._orig_columns = list(X.columns) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(X.shape[1])]

    def _infer_schema(self, X_df: pd.DataFrame):
        self._num_cols, self._cat_cols = [], []
        for c in X_df.columns:
            dt = X_df[c].dtype
            is_nonnumeric = (
                pd.api.types.is_object_dtype(dt)
                or pd.api.types.is_categorical_dtype(dt)
                or pd.api.types.is_bool_dtype(dt)
            )
            if is_nonnumeric:
                self._cat_cols.append(c)
            else:
                nunique = X_df[c].nunique(dropna=True)
                if nunique <= CAT_CARDINALITY_THRESHOLD:
                    self._cat_cols.append(c)
                else:
                    self._num_cols.append(c)
        self._cat_values = {}
        for c in self._cat_cols:
            # Only add an NA_CAT bucket if the column actually had missing values in
            # training -- otherwise it's a phantom class the decoder can sample from,
            # which decodes to None -> NaN and crashes the downstream classifier.
            col = X_df[c].astype("object").where(pd.notnull(X_df[c]), "NA_CAT")
            cats = sorted(map(str, pd.Index(col.unique().tolist()).tolist()))
            self._cat_values[c] = cats

    def _fit_transform_X(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (M, cat_targets) where M is the full [num | onehot] matrix used as
        VAE input/recon target for numerics, and cat_targets is (n, n_cat_cols) of
        integer class indices used for the cross-entropy loss."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self._orig_columns)
        Xc = X.copy()
        for c in Xc.columns:
            if c not in self._cat_cols:
                Xc[c] = pd.to_numeric(Xc[c], errors="coerce")

        mats: List[np.ndarray] = []
        self._feature_slices.clear()
        start = 0

        if self._num_cols:
            num_stack = []
            for c in self._num_cols:
                col = Xc[c].astype(float)
                med = float(np.nanmedian(col.values)) if np.isfinite(col).any() else 0.0
                self._num_min[c] = float(np.nanmin(col.values)) if np.isfinite(col).any() else med
                self._num_max[c] = float(np.nanmax(col.values)) if np.isfinite(col).any() else med
                colv = col.fillna(med).values.reshape(-1, 1)
                qt = QuantileTransformer(n_quantiles=min(self._num_quantiles, len(colv)),
                                          output_distribution="normal", subsample=int(1e6), random_state=7)
                Xt = qt.fit_transform(colv).astype(np.float32)
                self._num_tx[c] = qt
                num_stack.append(Xt)
            mat_num = np.concatenate(num_stack, axis=1).astype(np.float32)
            mats.append(mat_num)
            self._feature_slices["__NUM__"] = slice(start, start + mat_num.shape[1])
            start += mat_num.shape[1]

        cat_idx_cols = []
        for c in self._cat_cols:
            labels = self._cat_values[c]
            col = Xc[c].astype("object").where(pd.notnull(Xc[c]), "NA_CAT").astype(str)
            idxmap = {v: i for i, v in enumerate(labels)}
            idx = col.map(idxmap).fillna(0).astype(int).values
            onehot = np.zeros((len(Xc), len(labels)), dtype=np.float32)
            onehot[np.arange(len(Xc)), idx] = 1.0
            mats.append(onehot)
            self._feature_slices[c] = slice(start, start + onehot.shape[1])
            start += onehot.shape[1]
            cat_idx_cols.append(idx)

        M = np.concatenate(mats, axis=1) if mats else np.empty((len(Xc), 0), dtype=np.float32)
        cat_targets = np.stack(cat_idx_cols, axis=1).astype(np.int64) if cat_idx_cols else np.empty((len(Xc), 0), dtype=np.int64)
        return M.astype(np.float32), cat_targets

    def _inverse_transform(self, M: np.ndarray) -> pd.DataFrame:
        out: Dict[str, np.ndarray] = {}
        if "__NUM__" in self._feature_slices:
            sl = self._feature_slices["__NUM__"]
            num_block = M[:, sl]
            for i, c in enumerate(self._num_cols):
                qt = self._num_tx.get(c)
                x = num_block[:, i].reshape(-1, 1)
                try:
                    vals = qt.inverse_transform(x).reshape(-1)
                except Exception:
                    vals = x.reshape(-1)
                lo, hi = self._num_min.get(c, -np.inf), self._num_max.get(c, np.inf)
                out[c] = np.clip(vals, lo, hi)
        for c in self._cat_cols:
            sl = self._feature_slices[c]
            block = M[:, sl]
            idx = block.argmax(axis=1)
            labels = self._cat_values[c]
            out[c] = [None if labels[int(k)] == "NA_CAT" else labels[int(k)] for k in idx]
        cols = self._orig_columns if self._orig_columns else (self._num_cols + self._cat_cols)
        return pd.DataFrame({c: out[c] for c in cols})

    # ---- labels / conditioning (same convention as DPCTGAN v2) ----
    def _cond_vec(self, y: Optional[pd.Series], n: int) -> np.ndarray:
        if y is None:
            self._d_cond = 0
            self._has_y = False
            return np.zeros((n, 0), dtype=np.float32)
        yy = np.asarray(y).reshape(-1)
        classes = np.array(sorted(pd.unique(yy).tolist()))
        mapping = {v: i for i, v in enumerate(classes)}
        yy_mapped = np.vectorize(mapping.get)(yy).astype(int)
        d = int(yy_mapped.max() + 1)
        onehot = np.eye(d, dtype=np.float32)[yy_mapped]
        self._d_cond = d
        self._has_y = True
        self._y_classes = classes
        counts = np.bincount(yy_mapped, minlength=d).astype(float)
        self._y_probs = counts / counts.sum() if counts.sum() > 0 else np.ones(d) / d
        return onehot

    def _draw_cond_indices(self, n: int) -> np.ndarray:
        if not self._has_y or self._d_cond == 0:
            return np.zeros(n, dtype=int)
        probs = self._y_probs if self._y_probs is not None else np.ones(self._d_cond) / self._d_cond
        return np.random.choice(self._d_cond, size=n, p=probs)

    # ---- training ----
    def fit(self, X, y: Optional[pd.Series] = None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        self._remember_columns(X_df)
        self._infer_schema(X_df)
        X_mat, cat_targets = self._fit_transform_X(X_df)
        C_mat = self._cond_vec(y, X_mat.shape[0])

        device = torch.device("cuda" if (self.device == "auto" and torch.cuda.is_available()) else "cpu")
        self._device = device

        d_in = X_mat.shape[1]
        d_cond = C_mat.shape[1]
        model = _CVAE(d_in, d_cond, self.latent_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        X_tensor = torch.from_numpy(X_mat)
        C_tensor = torch.from_numpy(C_mat) if d_cond > 0 else torch.zeros((X_mat.shape[0], 0), dtype=torch.float32)
        Cat_tensor = torch.from_numpy(cat_targets)
        ds = TensorDataset(X_tensor, C_tensor, Cat_tensor)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        pe = PrivacyEngine(secure_mode=self.secure_mode)
        model, opt, dl = pe.make_private(
            module=model, optimizer=opt, data_loader=dl,
            noise_multiplier=self.noise_multiplier, max_grad_norm=self.max_grad_norm,
        )

        cat_slices = [self._feature_slices[c] for c in self._cat_cols]
        num_slice = self._feature_slices.get("__NUM__")

        model.train()
        for _epoch in range(self.epochs):
            for xb, cb, catb in dl:
                xb = xb.to(device).float()
                cb = cb.to(device).float()
                catb = catb.to(device).long()

                opt.zero_grad(set_to_none=True)
                x_hat, mu, logvar = model(xb, cb)

                loss = torch.zeros((), device=device)
                if num_slice is not None:
                    loss = loss + nn.functional.mse_loss(x_hat[:, num_slice], xb[:, num_slice], reduction="mean")
                for j, sl in enumerate(cat_slices):
                    logits = x_hat[:, sl]
                    target = catb[:, j]
                    loss = loss + nn.functional.cross_entropy(logits, target, reduction="mean")
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss + self.beta * kld

                loss.backward()
                opt.step()

        self._model = model
        self._fitted = True
        return self

    # ---- sampling ----
    def sample(self, n: int, return_y: bool = False, y_cond: Optional[np.ndarray] = None):
        assert self._fitted, "Call fit() before sample()."
        self._model.eval()

        if self._d_cond > 0:
            if y_cond is not None:
                y_cond = np.asarray(y_cond)
                if y_cond.shape[0] != n:
                    raise ValueError(f"y_cond length {y_cond.shape[0]} != n {n}")
                mapping = {v: i for i, v in enumerate(self._y_classes.tolist())}
                c_idx = np.vectorize(mapping.__getitem__)(y_cond).astype(int)
            else:
                c_idx = self._draw_cond_indices(n)
        else:
            c_idx = None

        with torch.no_grad():
            z = torch.randn(n, self.latent_dim, device=self._device)
            if self._d_cond > 0:
                C = torch.nn.functional.one_hot(torch.from_numpy(c_idx).to(self._device), num_classes=self._d_cond).float()
            else:
                C = torch.zeros((n, 0), device=self._device)
            # underlying module may be wrapped by Opacus GradSampleModule; .decode still resolves via getattr passthrough
            decode_fn = self._model.decode if hasattr(self._model, "decode") else self._model._module.decode
            Xgen = decode_fn(z, C).cpu().numpy().astype(np.float32)

        df = self._inverse_transform(Xgen)

        if return_y:
            y_out = self._y_classes[c_idx] if self._d_cond > 0 else None
            return df, y_out
        return df

    def sample_labels(self, n: int) -> Optional[np.ndarray]:
        if not self._has_y or self._y_classes is None:
            return None
        idx = self._draw_cond_indices(n)
        return self._y_classes[idx]
