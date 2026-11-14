"""Reuse batch statistics and running moments; add windowed summaries."""

import builtins
import warnings
import numpy as np
from numcompute.stats import mean, median, var, std, min, max, histogram, quantile, RunningStats
from numcompute._validation import real_array

__all__ = ["mean", "median", "var", "std", "min", "max", "histogram", "quantile", "RunningStats"]


class StreamingStats:
    """Track per-feature moments, exact quantiles and fixed-edge histograms.

    update_stats accepts numeric (n,d) or (n,) chunks and ignores NaNs.
    Infinity and changing feature counts raise ValueError. bins must contain
    increasing finite edges; values outside those edges are not counted.
    A sample window limits every result to the latest window_size rows.
    Without a window, exact quantiles retain all observations in memory.
    reset clears the observations and learned moments.
    """

    def __init__(self, *, bins=None, quantiles=(0.25, 0.5, 0.75), window_size=None):
        from numcompute._validation import integer

        self.window_size = None if window_size is None else integer(window_size, "window_size", 1)
        self.bins = np.asarray(np.linspace(-5, 5, 11) if bins is None else bins, dtype=float)
        if (
            self.bins.ndim != 1
            or self.bins.size < 2
            or not np.isfinite(self.bins).all()
            or np.any(np.diff(self.bins) <= 0)
        ):
            raise ValueError("bins must be strictly increasing finite edges")
        self.quantiles = np.asarray(quantiles, dtype=float)
        if (
            self.quantiles.ndim != 1
            or not np.isfinite(self.quantiles).all()
            or np.any((self.quantiles < 0) | (self.quantiles > 1))
        ):
            raise ValueError("quantiles must be a 1D array in [0,1]")
        self.reset()

    def reset(self):
        """Clear observations and moments, return self."""
        from collections import deque

        self._chunks = deque()
        self._rows = 0
        self.moments = RunningStats()
        return self

    def update_stats(self, X_chunk):
        """Update from copied (n,d) or (n,) observations and return self."""
        X = real_array(X_chunk, allow_nan=True)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2 or not X.shape[1]:
            raise ValueError("expected (n,) or (n,d) data")
        if self.moments.n_features is not None and X.shape[1] != self.moments.n_features:
            raise ValueError("feature count changed")
        self._chunks.append(X.copy())
        self._rows += len(X)
        if self.window_size is not None:
            while self._rows > self.window_size:
                remove = builtins.min(len(self._chunks[0]), self._rows - self.window_size)
                first = self._chunks.popleft()
                if remove < len(first):
                    self._chunks.appendleft(first[remove:].copy())
                self._rows -= remove
            self.moments = RunningStats().update(np.concatenate(self._chunks))
        else:
            self.moments.update(X)
        return self

    def result(self):
        """Return copies of count/mean/variance/quantiles/histograms/edges.

        Mean/variance/count have shape (d,), quantiles (q,d), histogram (d,b).
        Raises RuntimeError before any observations have been provided.
        """
        self.moments._require_data()
        X = np.concatenate(self._chunks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            qs = quantile(X, self.quantiles, axis=0, nan_policy="omit")
        hist = np.array([histogram(col, bins=self.bins)[0] for col in X.T])
        return {
            "count": self.moments.count,
            "mean": self.moments.mean,
            "variance": self.moments.var(),
            "min": self.moments.min,
            "max": self.moments.max,
            "quantiles": qs,
            "histogram": hist,
            "bin_edges": self.bins.copy(),
        }


__all__.append("StreamingStats")
