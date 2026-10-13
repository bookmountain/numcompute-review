"""Stable ordering, partial selection and binary search without input mutation."""

import numpy as np
from ._validation import real_array, integer


def stable_sort(values, axis=-1):
    """Return a stable ascending copy of values along axis (None flattens).

    Parameters: real array of any shape, axis (int or None). NaNs sort last.
    Returns: same shape, or (size,) if axis=None. Empty inputs are supported.
    Raises: TypeError for nonnumeric data, ValueError for infinity or bad axis.
    Complexity: O(n log n) time per axis slice, O(N) output/work space.
    """
    return np.sort(real_array(values, allow_nan=True, nonempty=False), axis=axis, kind="stable")


def multi_key_sort(data, columns, *, return_indices=False):
    """Sort rows lexicographically, first listed column most significant.

    Parameters: real data (n,d), nonempty sequence of unique column indices,
    return_indices (bool). Returns sorted (n,d) data, or (n,) row indices.
    NaNs sort last within a key. Equal rows retain input order.
    Raises: TypeError/ValueError for invalid input, duplicate or out-of-range keys.
    Complexity: O(p*n*log(n)) time for p keys, O(n*d) output space.
    """
    a = real_array(data, ndim=2, allow_nan=True, nonempty=False)
    keys = [integer(c, "column") for c in columns]
    if not keys or len(set(keys)) != len(keys) or max(keys) >= a.shape[1]:
        raise ValueError("columns must be unique, nonempty and in bounds")
    order = np.lexsort(tuple(a[:, c] for c in reversed(keys)))
    return order if return_indices else a[order]


def topk(values, k, largest=True, return_indices=True):
    """Select k finite values with argpartition, then sort the selected subset.

    Parameters: real values (n,), integer k in [0,n], largest (bool),
    return_indices (bool). Returns (selected_values, indices) when true, otherwise
    values only. Largest results descend; smallest ascend. Ties favour earlier
    input positions both at the selection boundary and within the output.
    Raises: TypeError for invalid dtype/k; ValueError for NaN, infinity or bad k.
    Complexity: O(n + k log k) time, O(n + k) auxiliary space.
    """
    a = real_array(values, ndim=1, nonempty=False)
    k = integer(k, "k")
    if k > a.size:
        raise ValueError("k must not exceed the array length")
    if k == 0:
        indices = np.empty(0, dtype=np.intp)
    else:
        boundary = a.size - k if largest else k - 1
        partition = np.argpartition(a, boundary)
        cutoff = a[partition[boundary]]
        better = np.flatnonzero(a > cutoff if largest else a < cutoff)
        tied = np.flatnonzero(a == cutoff)[:k - better.size]
        candidates = np.concatenate((better, tied))
        # Integer group IDs avoid negating unsigned values or losing precision.
        _, groups = np.unique(a[candidates], return_inverse=True)
        order = np.lexsort((candidates, -groups if largest else groups))
        indices = candidates[order]
    result = a[indices]
    return (result, indices) if return_indices else result


def quickselect(values, k):
    """Return the zero-based kth smallest finite value without fully sorting.

    Parameters: real array (n,), k in [0,n-1]. Returns a NumPy scalar.
    Uses iterative three-way partitioning; the input is never mutated.
    Raises: TypeError for invalid types; ValueError for empty/nonfinite data or k.
    Complexity: average O(n), worst O(n**2) time; O(n) working space.
    """
    work = real_array(values, ndim=1).copy()
    k = integer(k, "k")
    if k >= work.size:
        raise ValueError("k must be smaller than the array length")
    while True:
        pivot = work[work.size // 2]
        below = work[work < pivot]
        equal_count = np.count_nonzero(work == pivot)
        if k < below.size:
            work = below
        elif k < below.size + equal_count:
            return pivot
        else:
            k -= below.size + equal_count
            work = work[work > pivot]


def binary_search(sorted_array, x):
    """Return (left insertion index, exists) for finite scalar x in sorted (n,).

    Input must already be ascending; sortedness is deliberately not checked to
    preserve logarithmic search. Empty arrays return (0, False).
    Raises: TypeError for non-real types; ValueError for shape or nonfinite x.
    Complexity: O(log n) time and O(1) space for an existing NumPy array.
    Nonfinite array elements are outside this function's input contract.
    """
    a = np.asarray(sorted_array)
    if a.ndim != 1:
        raise ValueError("sorted_array must be one-dimensional")
    if a.dtype.kind not in "biuf":
        raise TypeError("sorted_array must be real numeric")
    target = real_array(x, ndim=0)
    lo, hi = 0, a.size
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo, bool(lo < a.size and a[lo] == target)
