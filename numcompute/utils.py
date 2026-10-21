"""Shared numerical utilities: activations, reductions, distances, and batches.

All utility functions are implemented using NumPy and the standard library.
Docstrings describe validation, edge cases, shapes, and memory requirements.

Except for the explicitly named logsumexp operation, public names and signatures
are provisional choices inferred from ASSIGNMENT_REQUIREMENTS.md. Numerical
utility functions accept real numeric NumPy arrays; iter_batches accepts NumPy
arrays of any dtype. Functions must not mutate their inputs. Floating-point numerical
outputs use float64; index outputs use np.intp. Negative axes follow NumPy
conventions. Non-contiguous arrays are supported. Complexity estimates include
output storage unless stated otherwise, with N denoting the input element count.
"""

from typing import Iterator, Optional

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the logistic activation without exponential overflow.

    Parameters
    ----------
    x : numpy.ndarray, shape (...,)
        Real integer or floating-point values of any shape, including a
        scalar array or an empty array. Boolean, complex, and object arrays
        are not accepted. The input is not modified.

    Returns
    -------
    numpy.ndarray, shape x.shape
        Float64 values in [0, 1]. NaNs propagate elementwise; negative
        and positive infinity map to 0 and 1 respectively.

    Raises
    ------
    TypeError
        If x is not a real numeric NumPy array.

    Notes
    -----
    Uses O(N) time and O(N) space. Sign-aware evaluation avoids exponential
    overflow for large-magnitude logits. Exponential underflow is harmless
    and yields the limiting activation value.
    """
    if not isinstance(x, np.ndarray) or x.dtype.kind not in "iuf":
        raise TypeError("x must be a real numeric NumPy array")

    values = np.asarray(x, dtype=np.result_type(x.dtype, np.float64))
    result = np.empty(x.shape, dtype=np.float64)
    nonnegative = values >= 0
    with np.errstate(under="ignore"):
        result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
        exp_values = np.exp(values[~nonnegative])
        result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


def relu(x: np.ndarray) -> np.ndarray:
    """Apply the rectified linear activation elementwise.

    Parameters
    ----------
    x : numpy.ndarray, shape (...,)
        Real integer or floating-point values; scalar arrays, empty arrays,
        and infinities are accepted. Boolean, complex, and object arrays are
        not accepted. The input is not modified.

    Returns
    -------
    numpy.ndarray, shape x.shape
        Float64 elementwise maximum of x and zero. NaNs propagate; negative
        infinity maps to zero and positive infinity is preserved.

    Raises
    ------
    TypeError
        If x is not a real numeric NumPy array.

    Notes
    -----
    Uses O(N) time and O(N) space.
    """
    if not isinstance(x, np.ndarray) or x.dtype.kind not in "iuf":
        raise TypeError("x must be a real numeric NumPy array")

    result = np.array(x, dtype=np.float64, copy=True)
    np.maximum(result, 0.0, out=result)
    return result


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert finite logits into probabilities along one axis stably.

    Parameters
    ----------
    x : numpy.ndarray, shape (d0, ..., dn)
        Finite real logits with at least one dimension.
    axis : int, default=-1
        Nonempty axis over which probabilities sum to one.

    Returns
    -------
    numpy.ndarray, shape x.shape
        Floating-point probabilities normalised independently in each slice.

    Raises
    ------
    TypeError
        If x is not a real numeric NumPy array or axis is not an integer.
    ValueError
        If x is scalar, axis is out of range or empty, or x contains NaN or
        infinity. Empty non-reduced dimensions are allowed.

    Notes
    -----
    Intended cost: O(N) time and O(N) space. Subtract each slice's maximum
    before exponentiation to avoid overflow for large finite logits.
    Integer inputs use temporary Python-integer object arrays for exact
    shifting before conversion to float64, preserving large-integer gaps.
    """
    if not isinstance(x, np.ndarray) or x.dtype.kind not in "iuf":
        raise TypeError("x must be a real numeric NumPy array")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer")
    if not -x.ndim <= axis < x.ndim:
        raise ValueError("axis is out of range")
    axis = int(axis) % x.ndim
    if x.shape[axis] == 0:
        raise ValueError("the reduced axis must be nonempty")
    if not np.isfinite(x).all():
        raise ValueError("x must contain only finite values")
    # Extreme finite differences may become -inf; their probability is zero.
    with np.errstate(over="ignore", under="ignore"):
        if x.dtype.kind in "iu":
            values = x.astype(object)
            shifted = np.asarray(
                values - np.max(values, axis=axis, keepdims=True), dtype=np.float64
            )
        else:
            values = np.asarray(x, dtype=np.result_type(x.dtype, np.float64))
            shifted = values - np.max(values, axis=axis, keepdims=True)
        weights = np.exp(shifted)
        probabilities = weights / np.sum(weights, axis=axis, keepdims=True)
        return np.asarray(probabilities, dtype=np.float64)


def logsumexp(
    x: np.ndarray, axis: Optional[int] = None, keepdims: bool = False
) -> np.ndarray:
    """Compute the logarithm of a sum of exponentials stably.

    Parameters
    ----------
    x : numpy.ndarray, shape (...,)
        Real values; scalar arrays, empty arrays, NaNs, and infinities allowed.
    axis : int or None, default=None
        Axis to reduce; None reduces all dimensions.
    keepdims : bool, default=False
        Retain reduced dimensions with length one.

    Returns
    -------
    numpy.ndarray
        Floating-point reduction. Shape is x.shape with the reduced axes
        removed or replaced by ones when keepdims=True. A full reduction
        without keepdims returns a zero-dimensional array. Empty or all
        negative-infinity slices return negative infinity. NaNs propagate
        per slice; otherwise any positive infinity gives positive infinity.

    Raises
    ------
    TypeError
        If x is not a real numeric NumPy array, axis is neither an integer
        nor None, or keepdims is not boolean.
    ValueError
        If axis is out of range; scalar arrays require axis=None.

    Notes
    -----
    Intended cost: O(N) time and O(N) working space plus reduction output.
    Use maximum-shifted exponentials and handle infinite/empty slices.
    Integer inputs use temporary Python-integer object arrays for exact
    shifting. Restore the integer maximum with its float64 rounding residual.
    """
    if not isinstance(x, np.ndarray) or x.dtype.kind not in "iuf":
        raise TypeError("x must be a real numeric NumPy array")
    if not isinstance(keepdims, (bool, np.bool_)):
        raise TypeError("keepdims must be boolean")
    if axis is not None:
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
            raise TypeError("axis must be an integer")
        if not -x.ndim <= axis < x.ndim:
            raise ValueError("axis is out of range")
        axis = int(axis) % x.ndim
    if x.dtype.kind in "iu" and x.size:
        values = x.astype(object)
        maxima = np.max(values, axis=axis, keepdims=True)
        shifted = np.asarray(values - maxima, dtype=np.float64)
        high = np.asarray(maxima, dtype=np.float64)
        # Python integers retain the part lost when the maximum is rounded.
        rounded_integers = np.frompyfunc(int, 1, 1)(high)
        residual = np.asarray(maxima - rounded_integers, dtype=np.float64)
        with np.errstate(under="ignore"):
            correction = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
            result = high + (residual + correction)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return np.asarray(result, dtype=np.float64)
    values = np.asarray(x, dtype=np.result_type(x.dtype, np.float64))
    maxima = np.max(values, axis=axis, keepdims=True, initial=-np.inf)
    finite = np.isfinite(maxima)
    shifted = np.full(values.shape, -np.inf, dtype=values.dtype)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        np.subtract(values, maxima, out=shifted, where=finite)
        totals = np.sum(np.exp(shifted), axis=axis, keepdims=True)
        # Non-finite maxima encode NaN, +inf, or empty/all--inf slices.
        result = np.where(finite, np.log(totals) + np.where(finite, maxima, 0), maxima)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return np.asarray(result, dtype=np.float64)


def euclidean_distance(
    x: np.ndarray, y: np.ndarray, axis: int = -1
) -> np.ndarray:
    """Compute Euclidean (L2) distances between corresponding vectors.

    Parameters
    ----------
    x, y : numpy.ndarray, identical shape (d0, ..., dn)
        Finite real arrays with at least one dimension. No broadcasting or
        all-pairs expansion is performed.
    axis : int, default=-1
        Coordinate axis to reduce; an empty coordinate axis gives zeros.

    Returns
    -------
    numpy.ndarray
        Floating-point distances with axis removed from x.shape. Inputs of
        shape (n_samples, n_features) give (n_samples,) with axis=-1;
        one-dimensional inputs give a zero-dimensional array.

    Raises
    ------
    TypeError
        If inputs are not real numeric NumPy arrays or axis is not an integer.
    ValueError
        If shapes differ, inputs are scalar or non-finite, or axis is invalid.

    Notes
    -----
    Intended cost: O(N) time and O(N) space; no quadratic pairwise matrix.
    Use a scaled norm to avoid unnecessary overflow from squaring differences.
    Two integer inputs are subtracted exactly using temporary Python-integer
    object arrays before float64 conversion, avoiding wraparound and lost gaps.
    Results outside the float64 range may still overflow to infinity.
    """
    if not isinstance(x, np.ndarray) or x.dtype.kind not in "iuf":
        raise TypeError("x must be a real numeric NumPy array")
    if not isinstance(y, np.ndarray) or y.dtype.kind not in "iuf":
        raise TypeError("y must be a real numeric NumPy array")
    if x.ndim == 0 or x.shape != y.shape:
        raise ValueError("x and y must have identical, non-scalar shapes")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer")
    if not -x.ndim <= axis < x.ndim:
        raise ValueError("axis is out of range")
    axis = int(axis) % x.ndim
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must contain only finite values")
    dtype = np.result_type(x.dtype, y.dtype, np.float64)
    with np.errstate(over="ignore", under="ignore"):
        if x.dtype.kind in "iu" and y.dtype.kind in "iu":
            differences = np.asarray(x.astype(object) - y.astype(object), dtype=np.float64)
        else:
            differences = np.asarray(x, dtype=dtype) - np.asarray(y, dtype=dtype)
        # hypot uses a scaled norm and avoids squaring large/small values.
        result = np.hypot.reduce(differences, axis=axis, initial=0.0)
        return np.asarray(result, dtype=np.float64)


def manhattan_distance(
    x: np.ndarray, y: np.ndarray, axis: int = -1
) -> np.ndarray:
    """Compute Manhattan (L1) distances between corresponding vectors.

    Parameters
    ----------
    x, y : numpy.ndarray, identical shape (d0, ..., dn)
        Finite real arrays with at least one dimension; no broadcasting.
    axis : int, default=-1
        Coordinate axis to reduce; an empty coordinate axis gives zeros.

    Returns
    -------
    numpy.ndarray
        Floating-point sums of absolute coordinate differences, with axis
        removed. Shape (n_samples, n_features) gives (n_samples,) for axis=-1;
        one-dimensional inputs give a zero-dimensional array.

    Raises
    ------
    TypeError
        If inputs are not real numeric NumPy arrays or axis is not an integer.
    ValueError
        If shapes differ, inputs are scalar or non-finite, or axis is invalid.

    Notes
    -----
    Intended cost: O(N) time and O(N) space; no quadratic pairwise matrix.
    Two integer inputs are subtracted exactly using temporary Python-integer
    object arrays before float64 conversion, avoiding wraparound and lost gaps.
    Other inputs are converted to float64 before subtraction.
    Results outside the float64 range may overflow to infinity.
    """
    if not isinstance(x, np.ndarray) or x.dtype.kind not in "iuf":
        raise TypeError("x must be a real numeric NumPy array")
    if not isinstance(y, np.ndarray) or y.dtype.kind not in "iuf":
        raise TypeError("y must be a real numeric NumPy array")
    if x.ndim == 0 or x.shape != y.shape:
        raise ValueError("x and y must have identical, non-scalar shapes")
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError("axis must be an integer")
    if not -x.ndim <= axis < x.ndim:
        raise ValueError("axis is out of range")
    axis = int(axis) % x.ndim
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must contain only finite values")
    with np.errstate(over="ignore", under="ignore"):
        if x.dtype.kind in "iu" and y.dtype.kind in "iu":
            differences = np.asarray(x.astype(object) - y.astype(object), dtype=np.float64)
        else:
            differences = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
        result = np.sum(np.abs(differences), axis=axis, dtype=np.float64)
        return np.asarray(result, dtype=np.float64)


def top_k_indices(x: np.ndarray, k: int, largest: bool = True) -> np.ndarray:
    """Select indexes of the k largest or smallest values deterministically.

    Parameters
    ----------
    x : numpy.ndarray, shape (n,)
        Finite real values; an empty array is allowed when k=0.
    k : int
        Number of indexes to return, satisfying 0 <= k <= n.
    largest : bool, default=True
        Select largest values when True, smallest values otherwise.

    Returns
    -------
    numpy.ndarray of dtype np.intp, shape (k,)
        Indexes ordered by decreasing value when largest=True, increasing
        otherwise. Equal values favour lower original indexes, including
        ties at the selection boundary. k=0 returns an empty index array.

    Raises
    ------
    TypeError
        If x is not a real numeric NumPy array, k is not an integer, or
        largest is not boolean.
    ValueError
        If x is not one-dimensional or contains NaN/infinity, or k is
        outside [0, n].

    Notes
    -----
    Intended cost: O(n + k log k) time and O(n + k) space using partitioning
    and deterministic boundary-tie handling. Selection semantics must remain
    consistent with the top-k functionality in sort_search.py, including
    ordering, tie-breaking, and invalid-input handling. Coordinate shared
    behaviour to avoid divergent selection semantics between the modules.
    """
    if not isinstance(x, np.ndarray) or x.dtype.kind not in "iuf":
        raise TypeError("x must be a real numeric NumPy array")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if not isinstance(largest, (bool, np.bool_)):
        raise TypeError("largest must be boolean")
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not np.isfinite(x).all():
        raise ValueError("x must contain only finite values")
    if not 0 <= k <= x.size:
        raise ValueError("k must satisfy 0 <= k <= len(x)")
    k = int(k)
    if k == 0:
        return np.empty(0, dtype=np.intp)
    values = np.asarray(x)
    boundary = values.size - k if largest else k - 1
    partition = np.argpartition(values, boundary)
    cutoff = values[partition[boundary]]
    better = np.flatnonzero(values > cutoff if largest else values < cutoff)
    ties = np.flatnonzero(values == cutoff)[:k - better.size]
    selected = np.concatenate((better, ties))
    keys = values[selected]
    if largest:
        # Complement reverses integer order without signed/unsigned overflow.
        keys = np.bitwise_not(keys) if keys.dtype.kind in "iu" else -keys
    order = np.lexsort((selected, keys))
    return np.asarray(selected[order], dtype=np.intp)


def iter_batches(
    x: np.ndarray, batch_size: int, drop_last: bool = False
) -> Iterator[np.ndarray]:
    """Iterate over consecutive batches along the first array dimension.

    Parameters
    ----------
    x : numpy.ndarray, shape (n_samples, ...)
        Array of any dtype with at least one dimension. Order is preserved.
    batch_size : int
        Positive maximum number of samples in each batch.
    drop_last : bool, default=False
        Discard the final batch if it contains fewer than batch_size samples.

    Yields
    ------
    numpy.ndarray, shape (b, ...)
        A view with 1 <= b <= batch_size and the same trailing dimensions
        and dtype as x. Empty inputs yield no batches. Mutating a yielded
        view may mutate x; iteration itself does not modify x.

    Raises
    ------
    TypeError
        If x is not a NumPy array, batch_size is not an integer, or drop_last
        is not boolean. Validation may occur when iteration begins.
    ValueError
        If x is scalar or batch_size is non-positive.

    Notes
    -----
    Intended cost: O(B) iteration time for B batches and O(1) auxiliary space
    for fixed array rank. Views avoid copying sample data; retained views
    keep the original array's storage alive. No shuffle index is allocated.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("x must be a NumPy array")
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError("batch_size must be an integer")
    if not isinstance(drop_last, (bool, np.bool_)):
        raise TypeError("drop_last must be boolean")
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch_size = int(batch_size)
    stop = x.shape[0]
    if drop_last:
        stop -= stop % batch_size
    for start in range(0, stop, batch_size):
        yield x[start:min(start + batch_size, stop)]
