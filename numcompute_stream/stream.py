"""Chunk orchestration with test-before-train evaluation and inspectable logs."""

import logging
import sys
from time import perf_counter

import numpy as np

from .metrics import StreamingMetrics, accuracy, _labels_array


def _memory_bytes(obj, seen=None):
    """Approximate owned Python/NumPy memory, without double-counting objects."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, np.ndarray):
        return size + (_memory_bytes(obj.base, seen) if obj.base is not None else 0)
    if isinstance(obj, dict):
        return size + sum(_memory_bytes(k, seen) + _memory_bytes(v, seen) for k, v in obj.items())
    if isinstance(obj, (tuple, list)):
        return size + sum(_memory_bytes(v, seen) for v in obj)
    if hasattr(obj, "__dict__"):
        return size + _memory_bytes(vars(obj), seen)
    return size


class StreamTrainer:
    """Manage an incremental model or pipeline and per-chunk evaluation history.

    fit_chunk evaluates the current model BEFORE training on that chunk. The
    initial untrained chunks are logged with NaN accuracy and never enter the
    cumulative denominator. score_chunk evaluates without fitting and adds its
    predictions to the same accumulator. Calling both on the same chunk counts
    it twice: use fit_chunk alone for progressive evaluation. metrics defaults
    to cumulative StreamingMetrics. A supplied windowed accumulator reports
    window metrics while cumulative_accuracy remains all-history accuracy.

    history records chunk, operation, n_samples, chunk_accuracy,
    cumulative_accuracy, evaluated_samples, fit_seconds, memory_bytes and ready.
    memory_bytes estimates model-owned Python/NumPy storage, not process RSS.
    logger is an optional standard-library Logger. One history record is kept
    per call. Invalid model or input errors propagate.
    """

    def __init__(self, model, *, metrics=None, logger=None):
        if not callable(getattr(model, "partial_fit", None)) or not callable(
            getattr(model, "predict", None)
        ):
            raise TypeError("model must provide partial_fit and predict")
        self.model = model
        self.metrics = StreamingMetrics() if metrics is None else metrics
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.history = []
        self._correct = self._evaluated = 0
        self._ready = bool(getattr(model, "is_fitted_", False))

    def _input(self, X, y):
        """Return X as (n, d) and y as (n,), checking that the rows match."""
        X = np.asarray(X)
        y = _labels_array(y, "y")
        if X.ndim != 2 or not X.shape[1] or X.shape[0] != y.size:
            raise ValueError("X must be (n,d), y (n,), with matching nonempty rows")
        return X, y

    def _evaluate(self, X, y):
        """Score the current model on a chunk, update metrics, return chunk accuracy."""
        pred = self.model.predict(X)
        scores = None
        if self.metrics.track_auc:
            estimator = self.model.steps[-1][1] if hasattr(self.model, "steps") else self.model
            match = np.flatnonzero(estimator.classes_ == self.metrics.pos_label)
            if not match.size:
                scores = np.zeros(len(X))
            else:
                scores = self.model.predict_proba(X)[:, match[0]]
        self.metrics.update(y, pred, scores)
        self._correct += int(np.count_nonzero(y == pred))
        self._evaluated += len(y)
        return float(accuracy(y, pred))

    def _record(self, X, operation, chunk_accuracy, elapsed):
        """Append one log record for this chunk and return a copy of it."""
        record = {
            "chunk": len(self.history) + 1,
            "operation": operation,
            "n_samples": len(X),
            "chunk_accuracy": chunk_accuracy,
            "cumulative_accuracy": (
                self._correct / self._evaluated if self._evaluated else float("nan")
            ),
            "evaluated_samples": self._evaluated,
            "fit_seconds": elapsed,
            "memory_bytes": _memory_bytes(self.model),
            "ready": bool(getattr(self.model, "is_fitted_", self._ready)),
        }
        self.history.append(record)
        self.logger.info("stream chunk: %s", record)
        return record.copy()

    def fit_chunk(self, X, y):
        """Test current model, then partial_fit on (n,d)/(n,) chunk; return log."""
        X, y = self._input(X, y)
        ready = getattr(self.model, "is_fitted_", self._ready)
        score = self._evaluate(X, y) if ready else float("nan")
        start = perf_counter()
        self.model.partial_fit(X.copy(), y.copy())
        self._ready = True
        return self._record(X, "fit", score, perf_counter() - start)

    def score_chunk(self, X, y):
        """Evaluate without training and return a log; raises if untrained."""
        X, y = self._input(X, y)
        score = self._evaluate(X, y)
        return self._record(X, "score", score, 0.0)
