"""Incremental, row-preserving preprocessing for (samples, features) arrays.

fit() resets learned state; partial_fit() updates it. Inputs are never modified.
NaNs are ignored during numeric estimation and preserved by scalers. Features
with no observations yet use neutral defaults (mean=0, scale=1, imputation=0).
Exact median imputation retains observations. Unsupported dtypes raise
TypeError; invalid shapes, feature changes and infinities raise ValueError.
"""

import numpy as np

from numcompute._validation import matrix
from numcompute.preprocessing import (
    StandardScaler as _StandardScaler,
    MinMaxScaler as _MinMaxScaler,
    SimpleImputer as _SimpleImputer,
    OneHotEncoder as _OneHotEncoder,
)
from .stats import RunningStats


class _NumericTransformer:
    """Shared fit and chunk checks for the incremental numeric transformers."""

    def fit(self, X):
        """Reuse batch fitting and validation, then seed incremental state."""
        super().fit(X)
        self.__dict__.pop("_stats", None)
        self.__dict__.pop("_values", None)
        return self.partial_fit(X)

    def _chunk(self, X):
        """Return X as a numeric 2D array with the same width as earlier chunks."""
        X = matrix(X)
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        self.n_features_in_ = X.shape[1]
        return X


class StandardScaler(_NumericTransformer, _StandardScaler):
    """Update column means and population variances as chunks arrive.

    partial_fit handles all-NaN chunks using neutral defaults until a finite
    observation arrives. Constant columns use scale one.
    """

    def partial_fit(self, X):
        """Update mean_/var_/scale_ from X (n,d); return self."""
        X = self._chunk(X)
        if not hasattr(self, "_stats"):
            self._stats = RunningStats()
        self._stats.update(X)
        self.n_samples_seen_ = self._stats.count
        self.mean_ = np.nan_to_num(self._stats.mean, nan=0.0)
        self.var_ = np.nan_to_num(self._stats.var(), nan=0.0)
        self.scale_ = np.where(self.var_ > 0, np.sqrt(self.var_), 1.0)
        return self


class MinMaxScaler(_NumericTransformer, _MinMaxScaler):
    """Running feature limits, with unit denominators for constant columns."""

    def partial_fit(self, X):
        """Update data_min_/data_max_/data_range_ from X (n,d); return self."""
        X = self._chunk(X)
        if not hasattr(self, "_stats"):
            self._stats = RunningStats()
        self._stats.update(X)
        self.data_min_ = np.nan_to_num(self._stats.min, nan=0.0)
        self.data_max_ = np.nan_to_num(self._stats.max, nan=0.0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.n_samples_seen_ = self._stats.count
        return self


class SimpleImputer(_NumericTransformer, _SimpleImputer):
    """Incremental mean, exact median or constant NaN replacement.

    Mean and constant imputation keep only per-column estimates. Exact median
    retains all chunks, so memory and update time grow with the stream.
    """

    def partial_fit(self, X):
        """Update replacement estimates from X (n,d); return self."""
        X = self._chunk(X)
        if self.strategy not in ("mean", "median"):
            return _SimpleImputer.fit(self, X)
        if self.strategy == "mean":
            if not hasattr(self, "_stats"):
                self._stats = RunningStats()
            self._stats.update(X)
            self.statistics_ = np.nan_to_num(self._stats.mean, nan=0.0)
        else:
            if not hasattr(self, "_values"):
                self._values = []
            self._values.append(X.copy())
            values = np.concatenate(self._values)
            self.statistics_ = np.array(
                [
                    np.median(col[~np.isnan(col)]) if np.isfinite(col).any() else 0.0
                    for col in values.T
                ]
            )
        return self


# "Imputer" is the class name used in Assignment2.2-specs.md.
Imputer = SimpleImputer


class OneHotEncoder(_OneHotEncoder):
    """Dense incremental categorical encoding with stable output column indexes.

    New (feature, category) pairs append columns; old columns never move, even
    when earlier input features acquire new categories. categories_ lists sorted
    values per input feature; output_features_ records the actual column order.
    X must be a homogeneous 2D string or finite numeric array. New categories
    increase output width. A downstream buffered tree requires fixed width,
    so its pipeline must receive the same category set across chunks.
    """

    def _input(self, X, training=True):
        """Validate a chunk and check it matches the width and kind seen before."""
        X = self._check_input(X, nonempty=training)
        if hasattr(self, "n_features_in_"):
            if X.shape[1] != self.n_features_in_:
                raise ValueError("feature count changed")
            if (X.dtype.kind in "US") != self._strings:
                raise TypeError("category kind must match previous chunks")
        return X

    def fit(self, X):
        """Reuse batch category discovery, then record stable output columns."""
        super().fit(X)
        self._strings = self._input_kind in "US"
        self.output_features_ = list(zip(self._feature_indices.tolist(), self._categories.tolist()))
        return self

    def partial_fit(self, X):
        """Append unseen categories from X (n,d), returning self."""
        if not hasattr(self, "n_features_in_"):
            return self.fit(X)
        X = self._input(X)
        for j in range(X.shape[1]):
            known = {v for f, v in self.output_features_ if f == j}
            self.output_features_.extend(
                (j, v.item()) for v in np.unique(X[:, j]) if v.item() not in known
            )
        self.categories_ = [
            np.asarray(sorted(v for f, v in self.output_features_ if f == j))
            for j in range(X.shape[1])
        ]
        self.n_features_out_ = len(self.output_features_)
        return self

    def transform(self, X):
        """Return (n,k) encoding; unknown categories raise unless set to ignore."""
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("call fit or partial_fit before transform")
        X = self._input(X, training=False)
        out = np.empty((len(X), self.n_features_out_), dtype=float)
        known = np.zeros(X.shape, dtype=bool)
        for col, (feature, category) in enumerate(self.output_features_):
            match = X[:, feature] == category
            out[:, col] = match
            known[:, feature] |= match
        if self.handle_unknown == "error" and not known.all():
            raise ValueError("input contains unknown categories")
        return out
