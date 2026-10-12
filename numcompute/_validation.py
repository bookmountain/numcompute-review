"""Internal array validation shared by the public modules."""

import operator
import numpy as np


def real_array(x, *, ndim=None, allow_nan=False, nonempty=True):
    """Validate a real numeric array without changing its dtype."""
    a = np.asarray(x)
    if a.dtype.kind not in "biuf":
        raise TypeError("expected a real numeric array")
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, got shape {a.shape}")
    if nonempty and a.size == 0:
        raise ValueError("array must not be empty")
    if np.isinf(a).any() or (not allow_nan and np.isnan(a).any()):
        raise ValueError("array contains unsupported NaN or infinite values")
    return a


def matrix(x, *, allow_nan=True, nonempty=True):
    """Validate a feature matrix (samples, features), converting to float64."""
    a = real_array(x, ndim=2, allow_nan=allow_nan, nonempty=nonempty)
    if a.shape[1] == 0:
        raise ValueError("at least one feature is required")
    return a.astype(float, copy=False)


def integer(value, name, minimum=0):
    """Reject booleans and fractional values in integer parameters."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def fitted(obj, x, *, allow_nan=True):
    """Check a fitted transformer's feature count; empty batches are allowed."""
    if not hasattr(obj, "n_features_in_"):
        raise RuntimeError("call fit before transform")
    a = matrix(x, allow_nan=allow_nan, nonempty=False)
    if a.shape[1] != obj.n_features_in_:
        raise ValueError(f"expected {obj.n_features_in_} features, got {a.shape[1]}")
    return a
