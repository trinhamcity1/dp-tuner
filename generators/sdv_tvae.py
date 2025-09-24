"""
TVAE baseline via SDV (MIT License).

Attribution:
- SDV (Synthetic Data Vault) project (MIT License): https://github.com/sdv-dev/SDV
- We use sdv.tabular.TVAE as a black-box baseline (non-DP) to set the utility target τ.

For DP-TVAE later, we will implement a custom training loop using DP-SGD (Opacus) and
ε accounting, since SDV's internal training loop is not DP-enabled by default.
"""
from typing import Optional
import pandas as pd

try:
    from sdv.tabular import TVAE
except Exception as e:  # pragma: no cover
    raise ImportError(
        "SDV not installed or failed to import. Install with `pip install sdv`.\n"
        f"Original error: {e}"
    )

from .base import BaseGenerator

class TvaeGenerator(BaseGenerator):
    def __init__(self,
                 epochs: int = 300,
                 batch_size: int = 500,
                 embedding_dim: int = 128,
                 compress_dims=(128, 128),
                 decompress_dims=(128, 128),
                 l2scale: float = 1e-5,
                 verbose: bool = False,
                 label_col: Optional[str] = None):
        self.label_col = label_col
        self.model = TVAE(
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            l2scale=l2scale,
            verbose=verbose,
        )
        self._fitted = False
        self._columns = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.label_col and self.label_col not in X.columns:
            if y is None:
                raise ValueError(
                    f"label_col='{self.label_col}' not in X and y is None. Provide y or set label_col=None."
                )
            X = X.copy()
            X[self.label_col] = y.values
        self._columns = list(X.columns)
        self.model.fit(X)
        self._fitted = True
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("TvaeGenerator.sample() called before fit().")
        out = self.model.sample(n)
        if self._columns is not None:
            out = out.reindex(columns=self._columns, fill_value=pd.NA)
        return out
