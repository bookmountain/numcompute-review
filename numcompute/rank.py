"""One-based rankings with explicit tie rules and NumPy-compatible percentiles."""

import numpy as np
from ._validation import real_array


def rank(data, method="average", *, nan_policy="omit"):
    """Rank real data (n,) using average, dense, or stable ordinal ties.

    Returns float ranks (n,), starting at 1. 'omit' leaves NaN positions NaN and
    ranks remaining values; 'propagate' makes all ranks NaN; 'raise' rejects NaNs.
    Empty arrays return empty ranks. Infinities are rejected.
    Raises: TypeError for dtype; ValueError for shape, policy or method.
    Complexity: O(n log n) time, O(n) working/output space.
    """
    if method not in {"average", "dense", "ordinal"}:
        raise ValueError("method must be 'average', 'dense', or 'ordinal'")
    if nan_policy not in {"omit", "propagate", "raise"}:
        raise ValueError("invalid nan_policy")
    a = real_array(data, ndim=1, allow_nan=nan_policy != "raise", nonempty=False)
    result = np.full(a.shape, np.nan)
    valid = ~np.isnan(a)
    if not valid.any() or (nan_policy == "propagate" and not valid.all()):
        return result
    positions = np.flatnonzero(valid)
    order = np.argsort(a[valid], kind="stable")
    ordered = a[valid][order]
    if method == "ordinal":
        ranked = np.arange(1, ordered.size + 1, dtype=float)
    else:
        starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        ends = np.r_[starts[1:], ordered.size]
        group = np.cumsum(np.r_[True, ordered[1:] != ordered[:-1]]) - 1
        values = (starts + 1 + ends) / 2 if method == "average" else np.arange(1, starts.size + 1)
        ranked = values[group]
    result[positions[order]] = ranked
    return result


def percentile(data, q, interpolation="linear", *, axis=None, nan_policy="omit", keepdims=False):
    """Return percentiles q in [0,100], scalar or vector, along axis.

    Parameters: real data, interpolation='linear'|'lower'|'higher'|'midpoint',
    axis (int/tuple/None), nan_policy='omit'|'propagate'|'raise', keepdims (bool).
    Returns a scalar for scalar q and all-axis reduction, otherwise an array;
    vector q adds a leading percentile dimension. NaN policies follow stats.
    Raises: TypeError for dtype; ValueError for empty data, invalid q/options.
    Complexity: O(N log N) upper-bound time and O(N) auxiliary space.
    """
    if nan_policy not in {"omit", "propagate", "raise"}:
        raise ValueError("nan_policy must be 'omit', 'propagate', or 'raise'")
    if interpolation not in {"linear", "lower", "higher", "midpoint"}:
        raise ValueError("invalid interpolation method")
    a = real_array(data, allow_nan=nan_policy != "raise")
    q = np.asarray(q, dtype=float)
    if q.ndim > 1 or not np.isfinite(q).all() or np.any((q < 0) | (q > 100)):
        raise ValueError("q must be a scalar or vector in [0, 100]")
    function = np.nanpercentile if nan_policy == "omit" else np.percentile
    return function(a, q, axis=axis, method=interpolation, keepdims=keepdims)
