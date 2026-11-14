"""Reuse batch metrics and add a single streaming accumulator."""

import numpy as np
from numcompute.metrics import (
    accuracy,
    precision,
    recall,
    f1,
    confusion_matrix,
    mse,
    roc_curve,
    auc,
    roc_auc_score,
    _AVERAGES,
    _check_zero_division,
    _confusion,
    _labels_array,
    _pair,
)
from numcompute._validation import real_array

__all__ = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "mse",
    "roc_curve",
    "auc",
    "roc_auc_score",
]


def _scores_from_cm(labels, cm, average, pos_label, zero_division):
    """Compute precision, recall and F1 from true-row confusion counts."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    if average == "binary":
        if len(labels) > 2:
            raise ValueError("average='binary' requires at most two classes")
        match = np.flatnonzero(labels == pos_label)
        if not match.size:
            raise ValueError("pos_label is not among the labels")
        tp, fp, fn = tp[match], fp[match], fn[match]
    elif average == "micro":
        tp, fp, fn = tp.sum(keepdims=True), fp.sum(keepdims=True), fn.sum(keepdims=True)
    scores = []
    for numerator, denominator in ((tp, tp + fp), (tp, tp + fn), (2 * tp, 2 * tp + fp + fn)):
        values = np.full(numerator.shape, zero_division, dtype=float)
        np.divide(numerator, denominator, out=values, where=denominator != 0)
        if average in ("binary", "micro"):
            values = values[0]
        elif average == "macro":
            values = values.mean()
        elif average == "weighted":
            values = np.sum(values * support) / support.sum()
        scores.append(values)
    return scores


class StreamingMetrics:
    """Accumulate classification scores, optionally over a sample window.

    update accepts matching nonempty (n,) label arrays. result returns accuracy,
    precision, recall, F1, a confusion matrix, labels and the sample count.
    average and zero_division follow the batch metrics; the default is macro.
    Invalid shapes, label types or options raise TypeError or ValueError.

    With track_auc=True, each update also needs finite positive-class scores.
    Exact binary AUC retains scores and labels; without a window this history
    grows with the stream. Otherwise only confusion counts are retained.
    Empty results and AUC with only one true class are NaN. reset clears state.
    """

    def __init__(
        self,
        *,
        labels=None,
        average="macro",
        pos_label=1,
        zero_division=0.0,
        window_size=None,
        track_auc=False,
    ):
        from numcompute._validation import integer

        if average not in _AVERAGES:
            raise ValueError("unsupported averaging method")
        self.average = average
        self.pos_label = pos_label
        self.zero_division = _check_zero_division(zero_division)
        self.window_size = None if window_size is None else integer(window_size, "window_size", 1)
        self.track_auc = bool(track_auc)
        self.labels = None if labels is None else _labels_array(labels, "labels").copy()
        if self.labels is not None and np.unique(self.labels).size != self.labels.size:
            raise ValueError("labels must not contain duplicates")
        self.reset()

    def reset(self):
        """Clear counts, score history and window; return self."""
        from collections import deque

        self._counts = {}
        self._observed = set()
        self._history = deque()
        self._size = 0
        self._strings = None
        return self

    def update(self, y_true, y_pred, y_score=None):
        """Add one chunk of labels, and its scores when track_auc=True; return self.

        y_true and y_pred are matching nonempty (n,) arrays. y_score holds
        positive-class scores and is required exactly when track_auc=True.
        An invalid chunk raises before any total is changed.
        """
        y_true, y_pred = _pair(y_true, y_pred)
        is_string = y_true.dtype.kind in "US"
        if self._strings is not None and is_string != self._strings:
            raise TypeError("label kind changed between chunks")
        classes, chunk_cm = _confusion(y_true, y_pred, self.labels)
        scores = self._check_scores(y_score, y_true, classes)
        self._strings = is_string
        self._observed.update(classes.tolist())
        self._add_counts(classes, chunk_cm, sign=1)
        self._size += len(y_true)
        if self.track_auc or self.window_size is not None:
            self._history.append((y_true.copy(), y_pred.copy(), scores))
        if self.window_size is not None:
            self._evict_outside_window()
        return self

    def _check_scores(self, y_score, y_true, classes):
        """Validate positive-class scores; return a copy, or None without AUC."""
        if not self.track_auc:
            if y_score is not None:
                raise ValueError("set track_auc=True to accumulate scores")
            return None
        if y_score is None:
            raise ValueError("track_auc requires y_score on every update")
        scores = real_array(y_score, ndim=1).copy()
        if scores.shape != y_true.shape:
            raise ValueError("scores must have the same shape as labels")
        if len(self._observed | set(classes.tolist())) > 2:
            raise ValueError("AUC requires at most two classes")
        return scores

    def _add_counts(self, classes, confusion, sign):
        """Add (sign=1) or remove (sign=-1) a chunk's confusion counts.

        Totals are kept per (true label, predicted label) pair, so new labels
        can appear mid-stream. Only the non-zero cells are visited.
        """
        for i, j in np.argwhere(confusion):
            key = (self._scalar(classes[i]), self._scalar(classes[j]))
            self._counts[key] = self._counts.get(key, 0) + sign * int(confusion[i, j])

    def _evict_outside_window(self):
        """Drop the oldest rows until at most window_size rows remain."""
        while self._size > self.window_size:
            old_true, old_pred, old_scores = self._history.popleft()
            n_drop = min(len(old_true), self._size - self.window_size)
            old_classes, old_confusion = _confusion(old_true[:n_drop], old_pred[:n_drop], None)
            self._add_counts(old_classes, old_confusion, sign=-1)
            if n_drop < len(old_true):  # keep the newer part of a partly dropped chunk
                self._history.appendleft(
                    (
                        old_true[n_drop:].copy(),
                        old_pred[n_drop:].copy(),
                        None if old_scores is None else old_scores[n_drop:].copy(),
                    )
                )
            self._size -= n_drop
        self._counts = {key: count for key, count in self._counts.items() if count}
        self._observed = {label for key in self._counts for label in key}

    @staticmethod
    def _scalar(value):
        """Turn a NumPy scalar into a plain Python value for use as a dict key."""
        return value.item() if isinstance(value, np.generic) else value

    def result(self):
        """Return independent metric values and true-row/predicted-column counts."""
        if self.labels is not None:
            labels = self.labels.copy()
        else:
            values = sorted(self._observed)
            labels = np.asarray(values)
            if labels.dtype.kind == "f" and all(isinstance(v, (int, bool)) for v in values):
                labels = np.asarray(values, dtype=object)
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
        lookup = {label: i for i, label in enumerate(labels.tolist())}
        for (a, b), count in self._counts.items():
            cm[lookup[a], lookup[b]] = count
        result = {"labels": labels, "confusion_matrix": cm, "n_samples": self._size}
        if not self._size:
            result.update(
                {name: np.float64(np.nan) for name in ("accuracy", "precision", "recall", "f1")}
            )
            if self.track_auc:
                result["roc_auc"] = np.float64(np.nan)
            return result
        result["accuracy"] = np.float64(np.trace(cm) / self._size)
        scores = _scores_from_cm(labels, cm, self.average, self.pos_label, self.zero_division)
        result.update(zip(("precision", "recall", "f1"), scores))
        if self.track_auc:
            truth = np.concatenate([a for a, _, _ in self._history])
            scores = np.concatenate([c for _, _, c in self._history])
            result["roc_auc"] = (
                roc_auc_score(truth, scores, pos_label=self.pos_label)
                if np.unique(truth).size == 2
                else np.float64(np.nan)
            )
        return result


__all__.append("StreamingMetrics")
