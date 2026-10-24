"""Classification and regression metrics with explicit label order and averaging.

Label conventions
-----------------
* ``y_true`` and ``y_pred`` are 1-D arrays of equal length holding class
  labels: integers, booleans or strings. Floating-point labels are accepted
  only when they are finite.
* ``labels=None`` means the sorted unique values of ``y_true`` and ``y_pred``
  combined. Passing ``labels`` fixes the class order explicitly (useful when a
  class is absent from a small test set); every value present in the data
  must appear in ``labels`` or ``ValueError`` is raised.
* In the confusion matrix, row ``i`` is the true class ``labels[i]`` and
  column ``j`` is the predicted class ``labels[j]``, so ``cm[i, j]`` counts
  samples of class ``i`` predicted as class ``j``. The diagonal holds the
  correct predictions.

Averaging (``precision``, ``recall``, ``f1``)
---------------------------------------------
``'binary'`` (default)
    The score of ``pos_label`` only; requires at most two classes.
``'micro'``
    Global counts of TP, FP and FN pooled over classes.
``'macro'``
    Unweighted mean of the per-class scores.
``'weighted'``
    Mean of the per-class scores weighted by true-class frequency (support).
``None``
    The per-class scores as an array in label order.

Undefined ratios
----------------
A ratio with a zero denominator (no predicted positives for precision, no
true positives for recall, both for F1) is replaced by ``zero_division``
(default ``0.0``; ``numpy.nan`` is allowed) and a ``RuntimeWarning`` is
emitted, so a silent 0/0 never masquerades as a real score.

All functions return NumPy floats/arrays, never modify their inputs, and run
in O(n) time for n samples unless stated otherwise.
"""

import warnings

import numpy as np

from ._validation import real_array

__all__ = ["confusion_matrix", "accuracy", "precision", "recall", "f1", "mse",
           "roc_curve", "auc", "roc_auc_score"]

_AVERAGES = ("binary", "micro", "macro", "weighted", None)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def _labels_array(y, name):
    a = np.asarray(y)
    if a.dtype.kind == "O":
        try:
            a = np.asarray(a.tolist())
        except Exception:  # pragma: no cover - defensive
            pass
    if a.dtype.kind not in "biufUS":
        raise TypeError(f"{name} must hold numeric, boolean or string labels")
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.size == 0:
        raise ValueError(f"{name} must not be empty")
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or infinite labels")
    return a


def _pair(y_true, y_pred):
    t = _labels_array(y_true, "y_true")
    p = _labels_array(y_pred, "y_pred")
    if t.shape != p.shape:
        raise ValueError(f"y_true and y_pred must have the same length, got {t.size} and {p.size}")
    if (t.dtype.kind in "US") != (p.dtype.kind in "US"):
        raise TypeError("y_true and y_pred must both be string labels or both numeric")
    return t, p


def _resolve_labels(t, p, labels):
    """Return the sorted label vector and integer codes for t and p."""
    present = np.unique(np.concatenate((t, p)))
    if labels is None:
        classes = present
    else:
        classes = np.asarray(labels)
        if classes.ndim != 1 or classes.size == 0:
            raise ValueError("labels must be a non-empty 1-D sequence")
        if np.unique(classes).size != classes.size:
            raise ValueError("labels must not contain duplicates")
        if (classes.dtype.kind in "US") != (present.dtype.kind in "US"):
            raise TypeError("labels must have the same kind (string/numeric) as the data")
        if not np.isin(present, classes).all():
            missing = present[~np.isin(present, classes)]
            raise ValueError(f"values {missing.tolist()} appear in the data but not in labels")
    order = np.argsort(classes, kind="stable")
    sorted_classes = classes[order]
    codes_t = order[np.searchsorted(sorted_classes, t)]
    codes_p = order[np.searchsorted(sorted_classes, p)]
    return classes, codes_t, codes_p


# --------------------------------------------------------------------------- #
# Confusion matrix and accuracy
# --------------------------------------------------------------------------- #

def confusion_matrix(y_true, y_pred, labels=None):
    """Count true-versus-predicted label pairs.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
        Class labels (integer, boolean or string).
    labels : array_like, shape (n_classes,) or None
        Class order for the rows and columns. ``None`` uses the sorted unique
        labels present in either array.

    Returns
    -------
    numpy.ndarray of int64, shape (n_classes, n_classes)
        ``cm[i, j]`` is the number of samples with true label ``labels[i]``
        and predicted label ``labels[j]``.

    Raises
    ------
    TypeError
        Unsupported label dtype, or mixed string/numeric labels.
    ValueError
        Empty or non-1-D input, length mismatch, NaN labels, duplicate
        ``labels``, or data values missing from ``labels``.

    Notes
    -----
    O(n log k) time for k classes (sorting the label set and a searchsorted
    lookup) and O(n + k²) space.
    """
    return _confusion(y_true, y_pred, labels)[1]


def _confusion(y_true, y_pred, labels):
    t, p = _pair(y_true, y_pred)
    classes, ct, cp = _resolve_labels(t, p, labels)
    k = classes.size
    cm = np.bincount(ct * k + cp, minlength=k * k).reshape(k, k).astype(np.int64)
    return classes, cm


def accuracy(y_true, y_pred):
    """Fraction of samples whose predicted label equals the true label.

    Returns a ``numpy.float64`` in ``[0, 1]``. Raises as
    :func:`confusion_matrix`. O(n) time and space.
    """
    t, p = _pair(y_true, y_pred)
    return np.float64(np.mean(t == p))


# --------------------------------------------------------------------------- #
# Precision / recall / F1
# --------------------------------------------------------------------------- #

def _safe_divide(numerator, denominator, zero_division, what):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    undefined = denominator == 0
    if undefined.any():
        warnings.warn(f"{what} is undefined (0/0) for at least one class; "
                      f"using zero_division={zero_division!r}", RuntimeWarning, stacklevel=4)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = numerator / denominator
    return np.where(undefined, zero_division, ratio)


def _prf_counts(y_true, y_pred, average, pos_label, labels):
    if average not in _AVERAGES:
        raise ValueError("average must be 'binary', 'micro', 'macro', 'weighted' or None")
    classes, cm = _confusion(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    if average == "binary":
        if classes.size > 2:
            raise ValueError("average='binary' requires at most two classes; "
                             "use 'micro', 'macro', 'weighted' or None")
        match = np.flatnonzero(classes == np.asarray(pos_label))
        if match.size == 0:
            raise ValueError(f"pos_label={pos_label!r} is not among the labels {classes.tolist()}")
        i = match[0]
        tp, fp, fn, support = tp[i:i + 1], fp[i:i + 1], fn[i:i + 1], support[i:i + 1]
    elif average == "micro":
        tp, fp, fn, support = tp.sum(keepdims=True), fp.sum(keepdims=True), fn.sum(keepdims=True), support.sum(keepdims=True)
    return tp, fp, fn, support


def _check_zero_division(zero_division):
    if isinstance(zero_division, bool) or not isinstance(zero_division, (int, float, np.floating, np.integer)):
        raise TypeError("zero_division must be a number (0.0, 1.0 or numpy.nan)")
    return float(zero_division)


def _finish(per_class, average, support):
    if average in ("binary", "micro"):
        return np.float64(per_class[0])
    if average == "macro":
        return np.float64(np.mean(per_class))
    if average == "weighted":
        total = support.sum()
        return np.float64(np.sum(per_class * support) / total) if total else np.float64(np.nan)
    return per_class


def precision(y_true, y_pred, *, average="binary", pos_label=1, labels=None, zero_division=0.0):
    """Precision ``TP / (TP + FP)``: how many predicted positives were right.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n_samples,)
    average : {'binary', 'micro', 'macro', 'weighted', None}
        See the module docstring. ``'binary'`` scores ``pos_label`` only.
    pos_label : label
        Positive class for ``average='binary'``; must be one of the labels.
    labels : array_like or None
        Explicit class order, as in :func:`confusion_matrix`.
    zero_division : float
        Value used, with a ``RuntimeWarning``, when ``TP + FP == 0``.

    Returns
    -------
    numpy.float64, or numpy.ndarray of shape (n_classes,) for ``average=None``.

    Raises
    ------
    TypeError
        As :func:`confusion_matrix`, or a non-numeric ``zero_division``.
    ValueError
        As :func:`confusion_matrix`, an unknown ``average``, ``'binary'`` with
        more than two classes, or ``pos_label`` not among the labels.

    Notes
    -----
    O(n log k) time, O(n + k²) space via the confusion matrix.
    """
    zd = _check_zero_division(zero_division)
    tp, fp, _, support = _prf_counts(y_true, y_pred, average, pos_label, labels)
    return _finish(_safe_divide(tp, tp + fp, zd, "precision"), average, support)


def recall(y_true, y_pred, *, average="binary", pos_label=1, labels=None, zero_division=0.0):
    """Recall ``TP / (TP + FN)``: how many actual positives were found.

    Parameters, return values and exceptions follow :func:`precision`; the
    undefined case is a class with no true samples (``TP + FN == 0``).
    """
    zd = _check_zero_division(zero_division)
    tp, _, fn, support = _prf_counts(y_true, y_pred, average, pos_label, labels)
    return _finish(_safe_divide(tp, tp + fn, zd, "recall"), average, support)


def f1(y_true, y_pred, *, average="binary", pos_label=1, labels=None, zero_division=0.0):
    """F1 score ``2·TP / (2·TP + FP + FN)``, the harmonic mean of precision and recall.

    Parameters, return values and exceptions follow :func:`precision`. The
    score is undefined only when a class has no true samples and no predicted
    samples (``TP = FP = FN = 0``); a class that is never predicted but does
    occur has ``F1 = 0`` without a warning, because recall is a well-defined 0.
    """
    zd = _check_zero_division(zero_division)
    tp, fp, fn, support = _prf_counts(y_true, y_pred, average, pos_label, labels)
    return _finish(_safe_divide(2 * tp, 2 * tp + fp + fn, zd, "f1"), average, support)


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #

def mse(y_true, y_pred, axis=None):
    """Mean squared error ``mean((y_true - y_pred)**2)`` along ``axis``.

    Parameters
    ----------
    y_true, y_pred : array_like of real numbers, same shape (...)
        Finite values; NaN or infinity raise ``ValueError``.
    axis : None, int or tuple of int
        ``None`` averages over every element and returns a scalar; an axis
        returns the per-slice error with that axis removed (e.g. ``axis=0`` on
        ``(n_samples, n_outputs)`` gives one error per output).

    Returns
    -------
    numpy.float64 or numpy.ndarray of float64

    Raises
    ------
    TypeError
        Non-numeric input.
    ValueError
        Empty input, shape mismatch, NaN or infinity.

    Notes
    -----
    O(N) time and space for N elements.
    """
    t = real_array(y_true).astype(float, copy=False)
    p = real_array(y_pred).astype(float, copy=False)
    if t.shape != p.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {t.shape} and {p.shape}")
    return np.mean((t - p) ** 2, axis=axis)


# --------------------------------------------------------------------------- #
# ROC and AUC (binary)
# --------------------------------------------------------------------------- #

def roc_curve(y_true, y_score, *, pos_label=1):
    """Receiver operating characteristic for a binary problem.

    Parameters
    ----------
    y_true : array_like, shape (n_samples,)
        Labels with exactly two distinct values, one of them ``pos_label``.
    y_score : array_like of real numbers, shape (n_samples,)
        Finite scores where larger means more likely positive.
    pos_label : label
        Which value of ``y_true`` counts as positive.

    Returns
    -------
    fpr : numpy.ndarray of float64, shape (n_thresholds,)
        False-positive rate, non-decreasing from 0 to 1.
    tpr : numpy.ndarray of float64, shape (n_thresholds,)
        True-positive rate, non-decreasing from 0 to 1.
    thresholds : numpy.ndarray of float64, shape (n_thresholds,)
        Strictly decreasing; ``thresholds[0]`` is ``+inf`` (predict nothing
        positive) and each later point predicts ``score >= threshold``.

    Raises
    ------
    TypeError
        Unsupported label or score dtype.
    ValueError
        Empty, non-1-D or mismatched inputs, non-finite scores, more than two
        classes, ``pos_label`` absent, or only one class present (the curve is
        undefined when there are no positives or no negatives).

    Notes
    -----
    Tied scores are collapsed into a single point, so the curve has one point
    per distinct score plus the origin: ``n_thresholds <= n_unique + 1``. This
    is the standard treatment and makes :func:`auc` of the result equal to the
    Mann–Whitney probability with ties counted as one half. O(n log n) time
    for the sort, O(n) space.
    """
    t = _labels_array(y_true, "y_true")
    s = real_array(y_score, ndim=1).astype(float, copy=False)
    if t.shape != s.shape:
        raise ValueError(f"y_true and y_score must have the same length, got {t.size} and {s.size}")
    classes = np.unique(t)
    if classes.size > 2:
        raise ValueError("roc_curve is defined for binary labels only")
    if not np.any(classes == np.asarray(pos_label)):
        raise ValueError(f"pos_label={pos_label!r} does not appear in y_true")
    positive = t == np.asarray(pos_label)
    n_pos = int(positive.sum())
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_curve needs at least one positive and one negative sample")
    order = np.argsort(-s, kind="stable")
    scores = s[order]
    positive = positive[order]
    # Index of the last sample in each run of tied scores.
    distinct = np.r_[np.flatnonzero(np.diff(scores)), scores.size - 1]
    tps = np.cumsum(positive)[distinct]
    fps = 1 + distinct - tps
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[np.inf, scores[distinct]]
    return fpr, tpr, thresholds


def auc(x, y):
    """Area under a curve given by points ``(x, y)`` using the trapezoid rule.

    Parameters
    ----------
    x : array_like of real numbers, shape (n_points,)
        Monotonic (non-decreasing or non-increasing) coordinates, n >= 2.
    y : array_like of real numbers, shape (n_points,)

    Returns
    -------
    numpy.float64
        Signed area; for :func:`roc_curve` output this is the ROC AUC.

    Raises
    ------
    TypeError
        Non-numeric input.
    ValueError
        Fewer than two points, length mismatch, non-monotonic ``x``, NaN or
        infinity.

    Notes
    -----
    O(n) time and space.
    """
    xs = real_array(x, ndim=1).astype(float, copy=False)
    ys = real_array(y, ndim=1).astype(float, copy=False)
    if xs.shape != ys.shape:
        raise ValueError("x and y must have the same length")
    if xs.size < 2:
        raise ValueError("at least two points are required to compute an area")
    dx = np.diff(xs)
    if not (np.all(dx >= 0) or np.all(dx <= 0)):
        raise ValueError("x must be monotonic")
    return np.float64(np.sum(dx * (ys[1:] + ys[:-1]) / 2.0))


def roc_auc_score(y_true, y_score, *, pos_label=1):
    """Area under the ROC curve: ``auc(*roc_curve(y_true, y_score)[:2])``.

    Equals the probability that a random positive outscores a random negative,
    with ties counted as one half. Parameters and exceptions follow
    :func:`roc_curve`. O(n log n) time, O(n) space.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    return auc(fpr, tpr)
