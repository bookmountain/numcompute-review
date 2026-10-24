"""One-based rankings with explicit tie rules and NumPy-compatible percentiles."""

import numpy as np
from ._validation import real_array


def rank(data, method="average", *, ascending=True, nan_policy="omit"):
    """Rank real data (n,) using average, dense, or stable ordinal ties.

    Returns float ranks (n,), starting at 1. Rank 1 is the smallest value when
    ascending=True and the largest value when ascending=False; ordinal ties
    favour earlier positions in both directions. 'omit' leaves NaN positions NaN and
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
    keys = a[valid]
    if not ascending:
        # Integer group IDs avoid negating unsigned values or losing precision.
        unique_values, groups = np.unique(keys, return_inverse=True)
        keys = -groups  # largest value -> most negative -> ranked first
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    if method == "ordinal":
        ranked = np.arange(1, ordered.size + 1, dtype=float)
    else:
        # True where a sorted value differs from the one before it (a new tie group).
        is_new_group = np.r_[True, ordered[1:] != ordered[:-1]]
        group_id = np.cumsum(is_new_group) - 1      # which group each sorted value is in
        group_start = np.flatnonzero(is_new_group)  # first sorted index of each group
        group_size = np.bincount(group_id)          # how many values in each group
        if method == "average":
            first_rank = group_start + 1
            last_rank = group_start + group_size
            group_rank = (first_rank + last_rank) / 2
        else:  # dense: groups are numbered 1, 2, 3, ...
            group_rank = np.arange(1, group_start.size + 1)
        ranked = group_rank[group_id]
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
    # Booleans pass validation but NumPy cannot interpolate them, so use float.
    a = real_array(data, allow_nan=nan_policy != "raise").astype(float, copy=False)
    q = np.asarray(q, dtype=float)
    if q.ndim > 1 or not np.isfinite(q).all() or np.any((q < 0) | (q > 100)):
        raise ValueError("q must be a scalar or vector in [0, 100]")
    function = np.nanpercentile if nan_policy == "omit" else np.percentile
    return function(a, q, axis=axis, method=interpolation, keepdims=keepdims)
