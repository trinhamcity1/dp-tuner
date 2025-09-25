"""
TVAE baseline via SDV.

Works with:
- SDV >= 1.0: sdv.single_table.TVAESynthesizer (preferred)
- SDV 0.18.x: sdv.tabular.TVAE (fallback)

Interface:
    gen = TvaeGenerator(epochs=..., batch_size=..., label_col="readmit_30d")
    gen.fit(X_df, y_series)         # X_df is RAW pandas.DataFrame; y optional
    syn = gen.sample(n)             # returns RAW DataFrame with original columns
"""

from typing import Optional
import pandas as pd

# Try the new single-table API first
_API = None
try:
    from sdv.single_table import TVAESynthesizer
    _API = "single_table"
except Exception:
    try:
        from sdv.tabular import TVAE  # old API
        _API = "tabular"
    except Exception as e:
        raise ImportError(
            "SDV not installed or failed to import.\n"
            "Install with: `pip install sdv`\n"
            f"Original error: {e}"
        )

# Helper to build metadata for single-table API across SDV versions
def _build_metadata(df: pd.DataFrame):
    from sdv.metadata import SingleTableMetadata
    md = SingleTableMetadata()
    md.detect_from_dataframe(df)
    return md



from .base import BaseGenerator


class TvaeGenerator(BaseGenerator):
    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        # Old API extras (ignored by new single-table API):
        embedding_dim: int = 128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale: float = 1e-5,
        verbose: bool = False,
        label_col: Optional[str] = None,
    ):
        self.label_col = label_col
        self._fitted = False
        self._columns = None

        # Stash params; construct model later in fit()
        self._params = dict(
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            embedding_dim=embedding_dim,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            l2scale=l2scale,
        )
        self.model = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if self.label_col and self.label_col not in X.columns:
            if y is None:
                raise ValueError(
                    f"label_col='{self.label_col}' not in X and y is None. "
                    "Provide y or set label_col=None."
                )
            X = X.copy()
            X[self.label_col] = y.values

        self._columns = list(X.columns)

        if _API == "single_table":
            md = _build_metadata(X)
            self.model = TVAESynthesizer(
                metadata=md,
                epochs=self._params["epochs"],
                batch_size=self._params["batch_size"],
                verbose=self._params["verbose"],
            )
            self.model.fit(X)
        else:
            # Old tabular API
            self.model = TVAE(
                epochs=self._params["epochs"],
                batch_size=self._params["batch_size"],
                embedding_dim=self._params["embedding_dim"],
                compress_dims=self._params["compress_dims"],
                decompress_dims=self._params["decompress_dims"],
                l2scale=self._params["l2scale"],
                verbose=self._params["verbose"],
            )
            self.model.fit(X)

        self._fitted = True
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("TvaeGenerator.sample() called before fit().")

        out = self.model.sample(num_rows=n) if _API == "single_table" else self.model.sample(n)
        if self._columns is not None:
            out = out.reindex(columns=self._columns, fill_value=pd.NA)
        return out
