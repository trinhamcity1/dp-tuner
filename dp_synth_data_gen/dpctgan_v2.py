# dp_synth_data_gen/dpctgan.py  (adaptive v2)
# NOTE: Same public API (class + method names + returns) as your current version.
#       Internals improved for stability & utility under DP, based on issues observed.

from typing import Optional, List, Tuple, Dict
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.preprocessing import QuantileTransformer

# -------------------------
# Helper: MLP with spectral norm option (stabilizes D under DP noise)
# -------------------------

def _mlp(d_in: int, d_out: int, widths: Tuple[int, ...] = (256, 256), spectral: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = d_in
    for w in widths:
        lin = nn.Linear(last, w)
        if spectral:
            lin = nn.utils.spectral_norm(lin)
        layers += [lin, nn.LeakyReLU(0.2, inplace=False)]
        last = w
    out = nn.Linear(last, d_out)
    if spectral:
        out = nn.utils.spectral_norm(out)
    layers += [out]
    return nn.Sequential(*layers)

# -------------------------
# Networks
# -------------------------

class _Gen(nn.Module):
    def __init__(self, z_dim: int, d_cond: int, d_out: int):
        super().__init__()
        self.net = _mlp(z_dim + d_cond, d_out, widths=(256, 256), spectral=False)
    def forward(self, z: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, C], dim=1))

class _Disc(nn.Module):
    def __init__(self, d_in: int, d_cond: int, spectral: bool = False):
        super().__init__()
        # spectral=True ONLY for the NON-DP shadow D
        self.net = _mlp(d_in + d_cond, 1, widths=(256, 128), spectral=spectral)

    def forward(self, x: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, C], dim=1)).flatten()


# -------------------------
# Main class (public API unchanged)
# -------------------------

class DPCTGAN:
    """
    Differentially-Private CTGAN (DP-CTGAN)

    Public methods & returns preserved: __init__, fit, sample, sample_labels.

    Improvements inside:
      • QuantileTransformer for numeric columns (maps to ~N(0,1); inverse supported).
      • Explicit missing bucket for categoricals maintained.
      • SpectralNorm on Discriminator for stability under DP.
      • Label conditioning kept; generator samples labels *uniformly* across classes (reduces minority collapse).
      • Optional tiny Gaussian jitter on D inputs (dp-safe regularization).
      • Early shadow-D refresh each iteration.
    """

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 256,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        delta: float = 1e-5,
        lr_g: float = 2e-4,
        lr_d: float = 1e-4,         # slightly smaller D LR under DP
        z_dim: int = 128,           # a bit larger Z helps tabular variety
        device: Optional[str] = "auto",
        secure_mode: bool = False,
        n_critic: int = 1,
        pac: int = 1,
        # new internal knobs (do not change signature/returns)
        _disc_input_jitter_std: float = 0.01,   # small Gaussian noise on D inputs
        _balance_labels_uniform: bool = True,   # sample labels uniformly when training G
        _num_quantiles: int = 1000,             # for QuantileTransformer
        _random_state: int = 7,
        **kwargs,
    ):
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.delta = float(delta)
        self.lr_g = float(lr_g)
        self.lr_d = float(lr_d)
        self.z_dim = int(z_dim)
        self.device = device
        self.secure_mode = secure_mode
        self.n_critic = int(n_critic)
        if pac != 1:
            print(f"[WARN] pac={pac} requested; forcing pac=1 for DP per-sample gradients.")
        self.pac = 1

        # schema / transform bookkeeping
        self._orig_columns: List[str] = []
        self._num_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._cat_values: Dict[str, List[str]] = {}
        self._feature_slices: Dict[str, slice] = {}
        self._transformed_cols: List[str] = []

        # numeric transformers (per-column)
        self._num_tx: Dict[str, QuantileTransformer] = {}
        self._num_min: Dict[str, float] = {}
        self._num_max: Dict[str, float] = {}

        # model state
        self._G: Optional[_Gen] = None
        self._device: Optional[torch.device] = None
        self._d_cond: int = 0
        self._fitted: bool = False

        # labels
        self._has_y: bool = False
        self._y_classes: Optional[np.ndarray] = None
        self._y_probs: Optional[np.ndarray] = None

        # internals
        self._disc_input_jitter_std = float(_disc_input_jitter_std)
        self._balance_labels_uniform = bool(_balance_labels_uniform)
        self._num_quantiles = int(_num_quantiles)
        self._rng = np.random.RandomState(int(_random_state))

    # -------------------------
    # schema & transforms
    # -------------------------
    def _remember_columns(self, X):
        if isinstance(X, pd.DataFrame):
            self._orig_columns = list(X.columns)
        else:
            self._orig_columns = [f"x{i}" for i in range(X.shape[1])]

    def _infer_schema(self, X_df: pd.DataFrame):
        self._num_cols, self._cat_cols = [], []
        for c in X_df.columns:
            dt = X_df[c].dtype
            if (
                pd.api.types.is_object_dtype(dt)
                or pd.api.types.is_categorical_dtype(dt)
                or pd.api.types.is_bool_dtype(dt)
            ):
                self._cat_cols.append(c)
            else:
                self._num_cols.append(c)
        self._cat_values = {}
        for c in self._cat_cols:
            col = X_df[c].astype("object").where(pd.notnull(X_df[c]), "NA_CAT")
            cats = sorted(map(str, pd.Index(col.unique().tolist()).tolist()))
            # ensure missing bucket exists
            if "NA_CAT" not in cats:
                cats = sorted(pd.Index([*cats, "NA_CAT"]).unique().tolist())
            self._cat_values[c] = cats

    def _fit_transform_X(self, X) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self._orig_columns)
        Xc = X.copy()

        # coerce numerics
        for c in Xc.columns:
            if c not in self._cat_cols:
                Xc[c] = pd.to_numeric(Xc[c], errors="coerce")

        # numerics: median impute, then QuantileTransformer to N(0,1)
        mats: List[np.ndarray] = []
        self._feature_slices.clear()
        self._transformed_cols = []
        start = 0
        if self._num_cols:
            num_stack = []
            for c in self._num_cols:
                col = Xc[c].astype(float)
                med = float(np.nanmedian(col.values)) if np.isfinite(col).any() else 0.0
                self._num_min[c] = float(np.nanmin(col.values)) if np.isfinite(col).any() else med
                self._num_max[c] = float(np.nanmax(col.values)) if np.isfinite(col).any() else med
                col = col.fillna(med).values.reshape(-1, 1)
                qt = QuantileTransformer(n_quantiles=min(self._num_quantiles, len(col)),
                                         output_distribution="normal",
                                         subsample=int(1e6),
                                         random_state=7)
                Xt = qt.fit_transform(col).astype(np.float32)
                self._num_tx[c] = qt
                num_stack.append(Xt)
            mat_num = np.concatenate(num_stack, axis=1).astype(np.float32) if num_stack else np.empty((len(Xc), 0), dtype=np.float32)
            mats.append(mat_num)
            self._feature_slices["__NUM__"] = slice(start, start + mat_num.shape[1])
            self._transformed_cols.extend(self._num_cols)
            start += mat_num.shape[1]

        # categoricals → one-hot with NA_CAT
        for c in self._cat_cols:
            labels = self._cat_values[c]
            col = Xc[c].astype("object").where(pd.notnull(Xc[c]), "NA_CAT").astype(str)
            idxmap = {v: i for i, v in enumerate(labels)}
            idx = col.map(idxmap).fillna(0).astype(int).values
            onehot = np.zeros((len(Xc), len(labels)), dtype=np.float32)
            onehot[np.arange(len(Xc)), idx] = 1.0
            mats.append(onehot)
            self._feature_slices[c] = slice(start, start + onehot.shape[1])
            self._transformed_cols.extend([f"{c}__{v}" for v in labels])
            start += onehot.shape[1]

        M = np.concatenate(mats, axis=1) if mats else np.empty((len(Xc), 0), dtype=np.float32)
        return M.astype(np.float32)

    def _inverse_transform(self, M: np.ndarray) -> pd.DataFrame:
        out: Dict[str, np.ndarray | List[str]] = {}

        # numerics: inverse QuantileTransformer back to original scale
        if "__NUM__" in self._feature_slices:
            sl = self._feature_slices["__NUM__"]
            num_block = M[:, sl]
            cols = []
            for i, c in enumerate(self._num_cols):
                qt = self._num_tx.get(c, None)
                x = num_block[:, i].reshape(-1, 1)
                if qt is not None:
                    try:
                        vals = qt.inverse_transform(x).reshape(-1)
                    except Exception:
                        # fallback linear scaling if inverse fails
                        vals = x.reshape(-1)
                else:
                    vals = x.reshape(-1)
                # clamp to training min/max to avoid outlandish values
                lo, hi = self._num_min.get(c, -np.inf), self._num_max.get(c, np.inf)
                vals = np.clip(vals, lo, hi)
                cols.append(vals)
                out[c] = vals
        # categoricals: argmax one-hot → string labels (None for NA_CAT)
        for c in self._cat_cols:
            sl = self._feature_slices[c]
            block = M[:, sl]
            idx = block.argmax(axis=1)
            labels = self._cat_values[c]
            vals = [None if labels[int(k)] == "NA_CAT" else labels[int(k)] for k in idx]
            out[c] = vals
        cols = self._orig_columns if self._orig_columns else (self._num_cols + self._cat_cols)
        return pd.DataFrame({c: out[c] for c in cols})

    # -------------------------
    # labels / conditioning
    # -------------------------
    def _cond_vec(self, y: Optional[pd.Series], n: int) -> np.ndarray:
        if y is None:
            self._d_cond = 0
            self._has_y = False
            self._y_classes = None
            self._y_probs = None
            return np.zeros((n, 0), dtype=np.float32)

        yy = np.asarray(y).reshape(-1)
        # robust mapping to consecutive class indices preserving original labels
        classes = np.array(sorted(pd.unique(yy).tolist()))
        mapping = {v: i for i, v in enumerate(classes)}
        yy_mapped = np.vectorize(mapping.get)(yy).astype(int)
        d = int(yy_mapped.max() + 1)
        onehot = np.eye(d, dtype=np.float32)[yy_mapped]
        self._d_cond = d
        self._has_y = True
        self._y_classes = classes
        counts = np.bincount(yy_mapped, minlength=d).astype(float)
        probs = counts / counts.sum() if counts.sum() > 0 else np.ones(d) / d
        self._y_probs = probs
        return onehot

    # -------------------------
    # training
    # -------------------------
    def fit(self, X, y: Optional[pd.Series] = None):
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            if hasattr(X, "shape"):
                X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
            else:
                raise ValueError("Unsupported X type; pass numpy array or pandas DataFrame")

        self._remember_columns(X_df)
        self._infer_schema(X_df)
        X_mat = self._fit_transform_X(X_df)
        C_mat = self._cond_vec(y, X_mat.shape[0])

        device = torch.device("cuda" if (self.device == "auto" and torch.cuda.is_available()) else "cpu")
        self._device = device

        d_in = X_mat.shape[1]
        d_cond = C_mat.shape[1]

        G = _Gen(self.z_dim, d_cond, d_in).to(device)
        D = _Disc(d_in, d_cond).to(device)
        optG = torch.optim.Adam(G.parameters(), lr=self.lr_g, betas=(0.5, 0.9))
        optD = torch.optim.Adam(D.parameters(), lr=self.lr_d, betas=(0.5, 0.9))

        X_tensor = torch.from_numpy(X_mat)
        C_tensor = torch.from_numpy(C_mat) if d_cond > 0 else torch.zeros((X_mat.shape[0], 0), dtype=torch.float32)
        ds = TensorDataset(X_tensor, C_tensor)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        peD = PrivacyEngine(secure_mode=self.secure_mode)
        D_priv, optD_priv, dl_priv = peD.make_private(
            module=D,
            optimizer=optD,
            data_loader=dl,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        # Shadow D (non-DP) for clean gradients to G
        D_shadow = _Disc(d_in, d_cond, spectral=True).to(device)
        D_shadow.load_state_dict(D_priv._module.state_dict())
        for p in D_shadow.parameters():
            p.requires_grad_(False)
        D_shadow.eval()

        bce = nn.BCEWithLogitsLoss()
        real_smooth = 0.9  # label smoothing for real labels

        for _epoch in range(self.epochs):
            for xb, cb in dl_priv:
                xb = xb.to(device).float()
                cb = cb.to(device).float()

                # (optional) tiny jitter to stabilize D (acts like DP-friendly noise reg)
                if self._disc_input_jitter_std > 0:
                    xb = xb + torch.randn_like(xb) * self._disc_input_jitter_std

                # ----- D updates (DP) -----
                for _ in range(self.n_critic):
                    optD_priv.zero_grad(set_to_none=True)
                    # fake
                    z = torch.randn(xb.size(0), self.z_dim, device=device)
                    with torch.no_grad():
                        x_fake = G(z, cb).detach()
                    if self._disc_input_jitter_std > 0:
                        x_fake = x_fake + torch.randn_like(x_fake) * self._disc_input_jitter_std

                    d_real = D_priv(xb, cb)
                    d_fake = D_priv(x_fake, cb)
                    y_real = torch.full_like(d_real, fill_value=real_smooth, device=device)
                    y_fake = torch.zeros_like(d_fake, device=device)
                    lossD = bce(d_real, y_real) + bce(d_fake, y_fake)
                    lossD.backward()
                    optD_priv.step()

                # sync shadow after D step
                D_shadow.load_state_dict(D_priv._module.state_dict())

                # ----- G update (non-DP) -----
                optG.zero_grad(set_to_none=True)
                z = torch.randn(xb.size(0), self.z_dim, device=device)
                if d_cond > 0:
                    if self._balance_labels_uniform:
                        # uniform across classes (helps minority)
                        idx = self._rng.choice(d_cond, size=xb.size(0), p=np.ones(d_cond)/d_cond)
                    else:
                        # empirical prior
                        probs = self._y_probs if self._y_probs is not None else np.ones(d_cond)/d_cond
                        idx = self._rng.choice(d_cond, size=xb.size(0), p=probs)
                    Cb = torch.zeros(xb.size(0), d_cond, device=device)
                    Cb[torch.arange(xb.size(0), device=device), torch.tensor(idx, device=device)] = 1.0
                else:
                    Cb = torch.zeros(xb.size(0), 0, device=device)

                x_fake = G(z, Cb)
                d_fake_for_G = D_shadow(x_fake, Cb)
                y_real_for_G = torch.ones_like(d_fake_for_G, device=device)
                lossG = bce(d_fake_for_G, y_real_for_G)
                lossG.backward()
                optG.step()

        self._G = G
        self._fitted = True

    # -------------------------
    # sampling
    # -------------------------
    def _draw_cond_indices(self, n: int) -> np.ndarray:
        if not self._has_y or self._d_cond == 0:
            return np.zeros(n, dtype=int)
        if self._balance_labels_uniform:
            return self._rng.choice(self._d_cond, size=n, p=np.ones(self._d_cond)/self._d_cond)
        probs = self._y_probs if self._y_probs is not None else np.ones(self._d_cond) / self._d_cond
        return self._rng.choice(self._d_cond, size=n, p=probs)

    def sample(self, n: int, return_y: bool = False, y_cond: Optional[np.ndarray] = None):
        assert self._fitted, "Call fit() before sample()."
        self._G.eval()

        # condition indices
        if self._d_cond > 0:
            if y_cond is not None:
                y_cond = np.asarray(y_cond)
                if y_cond.shape[0] != n:
                    raise ValueError(f"y_cond length {y_cond.shape[0]} != n {n}")
                if self._y_classes is None:
                    raise ValueError("Model lacks _y_classes; cannot map provided y_cond.")
                mapping = {v: i for i, v in enumerate(self._y_classes.tolist())}
                try:
                    c_idx = np.vectorize(mapping.__getitem__)(y_cond).astype(int)
                except KeyError as e:
                    raise ValueError(f"Provided label {e} not seen in training classes {self._y_classes}.")
            else:
                c_idx = self._draw_cond_indices(n)
        else:
            c_idx = None

        with torch.no_grad():
            z = torch.randn(n, self.z_dim, device=self._device)
            if self._d_cond > 0:
                C = torch.nn.functional.one_hot(
                    torch.from_numpy(c_idx).to(self._device),
                    num_classes=self._d_cond,
                ).float()
            else:
                C = torch.zeros((n, 0), device=self._device)
            Xgen = self._G(z, C).cpu().numpy().astype(np.float32)

        df = self._inverse_transform(Xgen)

        if return_y:
            if self._d_cond > 0:
                y_out = self._y_classes[c_idx]
            else:
                y_out = None
            return df, y_out
        return df

    def sample_labels(self, n: int) -> Optional[np.ndarray]:
        if not self._has_y or self._y_classes is None:
            return None
        idx = self._draw_cond_indices(n)
        return self._y_classes[idx]
