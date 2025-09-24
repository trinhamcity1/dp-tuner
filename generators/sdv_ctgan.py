"""
CTGAN baseline via SDV (MIT License).

Attribution:
- SDV (Synthetic Data Vault) project (MIT License): https://github.com/sdv-dev/SDV
- We use sdv.tabular.CTGAN as a black-box baseline (non-DP) to set the utility target τ.

Note: For DP-CTGAN later, we cannot just import; we will need a custom/forked training loop
that supports per-sample gradient clipping + noise (DP-SGD) and ε accounting (Opacus).
"""
from typing import Optional
import pandas as pd

try:
    from sdv.tabular import CTGAN
except Exception as e:  # pragma: no cover
    raise ImportError(
        "SDV not installed or failed to import. Install with `pip install sdv`.\n"
        f"Original error: {e}"
    )

from .base import BaseGenerator

class CtganGenerator(BaseGenerator):
    def __init__(self,
                 epochs: int = 300,
                 batch_size: int = 500,
                 discriminator_steps: int = 5,
                 generator_lr: float = 2e-4,
                 discriminator_lr: float = 2e-4,
                 verbose: bool = False,
                 label_col: Optional[str] = None):
        self.label_col = label_col
        self.model = CTGAN(
            epochs=epochs,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            generator_lr=generator_lr,
            discriminator_lr=discriminator_lr,
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
            raise RuntimeError("CtganGenerator.sample() called before fit().")
        out = self.model.sample(n)
        # ensure original column order if SDV reorders
        if self._columns is not None:
            out = out.reindex(columns=self._columns, fill_value=pd.NA)
        return out