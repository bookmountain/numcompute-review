"""Axis-aware descriptive statistics, histograms, quantiles and streaming moments.

Every reduction accepts real numeric input (bool, integer or float) of any
shape and reduces along ``axis`` with the same output shapes as NumPy:
``axis=None`` reduces everything to a scalar, an integer or tuple of axes
removes those dimensions, and ``keepdims=True`` keeps them with length one.
Inputs are never modified and non-contiguous views are accepted.

NaN policy (shared by every function that takes ``nan_policy``):

``'propagate'`` (default)
    Any NaN in a slice makes that slice's result NaN, as in plain NumPy.
``'omit'``
    NaNs are ignored. A slice containing only NaNs gives NaN and emits a
    ``RuntimeWarning`` (``min``/``max``/``median`` included); ``histogram``
    counts nothing for such input.
``'raise'``
    A NaN anywhere raises ``ValueError`` before any computation.

Infinite values are rejected with ``ValueError`` by every function, and empty
input (``size == 0``) is rejected with ``ValueError``, so no result is ever
computed from zero observations. Complexity notes use N for the number of
input elements.
"""

import warnings

import numpy as np

from ._validation import real_array

__all__ = ["mean", "median", "var", "std", "min", "max", "histogram",
           "quantile", "RunningStats"]

_POLICIES = ("propagate", "omit", "raise")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _check_policy(nan_policy):
    if nan_policy not in _POLICIES:
        raise ValueError("nan_policy must be 'propagate', 'omit', or 'raise'")


def _prepare(data, nan_policy):
    """Validate input for a reduction and return the array."""
    _check_policy(nan_policy)
    return real_array(data, allow_nan=nan_policy != "raise")


def _warn_all_nan(a, axis):
    """Emit one RuntimeWarning if any slice along axis is entirely NaN."""
    if a.dtype.kind == "f" and np.isnan(a).all(axis=axis).any():
        warnings.warn("all-NaN slice encountered; result is NaN", RuntimeWarning,
                      stacklevel=3)


def _reduce(a, axis, keepdims, nan_policy, plain, omitting, **kwargs):
    """Apply a NumPy reduction or its nan-aware twin to a validated array."""
    if nan_policy != "omit":
        return plain(a, axis=axis, keepdims=keepdims, **kwargs)
    _warn_all_nan(a, axis)
    with warnings.catch_warnings():
        # NumPy's own all-NaN / empty-slice warnings are replaced by the
        # single consistent warning raised above.
        warnings.simplefilter("ignore", RuntimeWarning)
        return omitting(a, axis=axis, keepdims=keepdims, **kwargs)


# --------------------------------------------------------------------------- #
# Descriptive statistics
# --------------------------------------------------------------------------- #

def mean(data, axis=None, *, nan_policy="propagate", keepdims=False):
    """Arithmetic mean along ``axis``.

    Parameters
    ----------
    data : array_like of real numbers, shape (...)
    axis : None, int or tuple of int
    nan_policy : {'propagate', 'omit', 'raise'}
    keepdims : bool

    Returns
    -------
    numpy.float64 or numpy.ndarray of float64
        Scalar for ``axis=None`` without ``keepdims``; otherwise the input
        shape with the reduced axes removed (or kept with length one).

    Raises
    ------
    TypeError
        Non-numeric input.
    ValueError
        Empty input, infinities, NaN under ``'raise'``, or a bad policy/axis.

    Notes
    -----
    O(N) time, O(N) auxiliary space under ``'omit'`` (a NaN mask), O(1)
    otherwise beyond the output.
    """
    return _reduce(_prepare(data, nan_policy), axis, keepdims, nan_policy, np.mean, np.nanmean)


def median(data, axis=None, *, nan_policy="propagate", keepdims=False):
    """Median along ``axis`` (midpoint of the two central values for even n).

    Parameters, return shape and exceptions follow :func:`mean`. Boolean input
    is treated as 0/1. O(N) average time via introselect, O(N) space for the
    partitioned copy.
    """
    return _reduce(_prepare(data, nan_policy), axis, keepdims, nan_policy, np.median, np.nanmedian)


def var(data, axis=None, *, ddof=0, nan_policy="propagate", keepdims=False):
    """Variance along ``axis`` with ``ddof`` delta degrees of freedom.

    ``ddof=0`` gives the population variance (NumPy default); ``ddof=1`` the
    unbiased sample variance. Under ``'omit'`` the divisor is the count of
    non-NaN values in each slice minus ``ddof``; a slice with no more than
    ``ddof`` valid values gives NaN with a ``RuntimeWarning``.

    Raises
    ------
    TypeError
        Non-numeric input or non-integer ``ddof``.
    ValueError
        As :func:`mean`, or negative ``ddof``.

    Notes
    -----
    Two-pass (mean, then squared deviations): O(N) time, O(N) space.
    """
    if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)):
        raise TypeError("ddof must be an integer")
    if ddof < 0:
        raise ValueError("ddof must be >= 0")
    a = _prepare(data, nan_policy)
    if nan_policy != "omit":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = np.var(a, axis=axis, ddof=ddof, keepdims=keepdims)
        if np.isnan(result).any() and not np.isnan(a).any():
            warnings.warn("fewer observations than ddof; variance is NaN",
                          RuntimeWarning, stacklevel=2)
        return result
    valid = np.sum(~np.isnan(a), axis=axis, keepdims=keepdims)
    if np.any(valid <= ddof):
        warnings.warn("all-NaN slice or fewer valid observations than ddof; "
                      "variance is NaN", RuntimeWarning, stacklevel=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanvar(a, axis=axis, ddof=ddof, keepdims=keepdims)


def std(data, axis=None, *, ddof=0, nan_policy="propagate", keepdims=False):
    """Standard deviation: the square root of :func:`var` with the same arguments."""
    return np.sqrt(var(data, axis, ddof=ddof, nan_policy=nan_policy, keepdims=keepdims))


def min(data, axis=None, *, nan_policy="propagate", keepdims=False):  # noqa: A001
    """Minimum along ``axis``; the input dtype is preserved for ints and bools.

    Parameters, return shape and exceptions follow :func:`mean`. Under
    ``'omit'`` an all-NaN slice gives NaN with a ``RuntimeWarning``.
    O(N) time, O(1) auxiliary space (O(N) under ``'omit'``).
    """
    return _reduce(_prepare(data, nan_policy), axis, keepdims, nan_policy, np.amin, np.nanmin)


def max(data, axis=None, *, nan_policy="propagate", keepdims=False):  # noqa: A001
    """Maximum along ``axis``; see :func:`min`."""
    return _reduce(_prepare(data, nan_policy), axis, keepdims, nan_policy, np.amax, np.nanmax)


# --------------------------------------------------------------------------- #
# Histogram and quantiles
# --------------------------------------------------------------------------- #

def histogram(data, bins=10, *, range=None, density=False, nan_policy="omit"):  # noqa: A002
    """Histogram of the flattened input.

    Parameters
    ----------
    data : array_like of real numbers, any shape (flattened before binning)
    bins : int >= 1 or 1-D array_like of monotonically increasing edges
    range : (float, float) or None
        Lower and upper bin range; values outside are ignored. Defaults to
        ``(min, max)`` of the counted values, or ``(0, 1)`` when nothing is
        counted. Ignored when ``bins`` is an edge array.
    density : bool
        If True, counts are normalised so the integral over the range is 1.
    nan_policy : {'omit', 'raise', 'propagate'}
        ``'omit'`` (default) drops NaNs before counting and warns if nothing
        remains; ``'propagate'`` returns a float count array that is all NaN
        when any NaN is present (edges still computed from the finite
        values); ``'raise'`` rejects NaN.

    Returns
    -------
    counts : numpy.ndarray, shape (n_bins,)
        int64 (float64 when ``density=True`` or under ``'propagate'`` NaN).
    edges : numpy.ndarray, shape (n_bins + 1,)
        float64 bin edges; the last bin is closed on the right like NumPy.

    Raises
    ------
    TypeError
        Non-numeric input or non-integer ``bins`` count.
    ValueError
        Empty input, infinities, non-increasing edges, ``bins < 1``, bad
        ``range`` or policy, NaN under ``'raise'``.

    Notes
    -----
    O(N log b) time for b bins (NumPy uses searchsorted for uneven edges,
    O(N) for uniform ones), O(N + b) space.
    """
    a = _prepare(data, nan_policy).ravel()
    if np.ndim(bins) == 0:
        if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)):
            raise TypeError("bins must be an integer or a 1-D array of edges")
        if bins < 1:
            raise ValueError("bins must be >= 1")
    else:
        bins = real_array(bins, ndim=1, nonempty=True).astype(float)
        if bins.size < 2 or np.any(np.diff(bins) <= 0):
            raise ValueError("bin edges must be strictly increasing with length >= 2")
    if range is not None:
        lo, hi = (float(v) for v in range)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
            raise ValueError("range must be finite with range[0] < range[1]")
        range = (lo, hi)

    has_nan = a.dtype.kind == "f" and np.isnan(a).any()
    values = a[~np.isnan(a)] if has_nan else a
    if values.size == 0:
        if nan_policy == "omit":
            warnings.warn("all-NaN input; histogram counts are zero", RuntimeWarning,
                          stacklevel=2)
    counts, edges = np.histogram(values.astype(float, copy=False), bins=bins,
                                 range=range, density=density)
    if nan_policy == "propagate" and has_nan:
        counts = np.full(counts.shape, np.nan)
    return counts, edges


def quantile(data, q, axis=None, *, method="linear", nan_policy="propagate",
             keepdims=False):
    """Quantiles ``q`` in ``[0, 1]`` along ``axis``.

    Parameters
    ----------
    data : array_like of real numbers, shape (...)
    q : float or 1-D array_like of floats in [0, 1]
    axis : None, int or tuple of int
    method : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        Interpolation between order statistics, as in :func:`numpy.quantile`.
    nan_policy : {'propagate', 'omit', 'raise'}
    keepdims : bool

    Returns
    -------
    numpy.float64 or numpy.ndarray
        For scalar ``q`` the shape follows :func:`mean`; for a vector ``q`` of
        length m a leading axis of length m is added.

    Raises
    ------
    TypeError
        Non-numeric input.
    ValueError
        As :func:`mean`, or ``q`` outside ``[0, 1]`` / not 0-D or 1-D, or an
        unknown ``method``.

    Notes
    -----
    O(N log N) worst-case time (partial sort per slice), O(N) space. For
    ``q=0.5`` this equals :func:`median`.
    """
    if method not in {"linear", "lower", "higher", "midpoint", "nearest"}:
        raise ValueError("method must be 'linear', 'lower', 'higher', 'midpoint' or 'nearest'")
    q = np.asarray(q, dtype=float)
    if q.ndim > 1 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("q must be a scalar or 1-D vector of values in [0, 1]")
    a = _prepare(data, nan_policy).astype(float, copy=False)
    return _reduce(a, axis, keepdims, nan_policy, np.quantile, np.nanquantile,
                   q=q, method=method)


# --------------------------------------------------------------------------- #
# Streaming statistics
# --------------------------------------------------------------------------- #

class RunningStats:
    """Mergeable per-column streaming mean, variance, min and max.

    Observations are accumulated with Welford's single-pass update, and two
    accumulators (or an accumulator and a batch) are combined with the
    Chan–Golub–LeVeque parallel formula, so results are identical whether the
    data arrive in one batch, many uneven chunks, or are merged from separate
    objects. Each feature column is tracked independently.

    Parameters
    ----------
    nan_policy : {'omit', 'propagate', 'raise'}
        ``'omit'`` (default) skips NaN observations per column, so columns can
        have different counts; ``'propagate'`` makes every statistic of a
        column NaN once it has seen a NaN; ``'raise'`` rejects NaN input.

    Attributes
    ----------
    n_features : int or None
        Number of columns, fixed by the first non-empty ``update``.
    count : numpy.ndarray of int64, shape (n_features,)
        Observations counted per column (NaNs excluded under ``'omit'``).
    mean, min, max : numpy.ndarray of float64, shape (n_features,)
        NaN for a column with no observations.

    Notes
    -----
    Batches of shape ``(n,)`` are n observations of one feature; batches of
    shape ``(n, d)`` are n observations of d features. ``update`` costs
    O(n*d) time and space for the batch; ``merge`` and the statistics are
    O(d). Infinite values are rejected.
    """

    def __init__(self, *, nan_policy="omit"):
        _check_policy(nan_policy)
        self.nan_policy = nan_policy
        self.n_features = None
        self._count = None
        self._mean = None
        self._m2 = None
        self._min = None
        self._max = None
        self._poisoned = None

    # -- construction helpers ------------------------------------------------ #

    def _allocate(self, d):
        self.n_features = d
        self._count = np.zeros(d, dtype=np.int64)
        self._mean = np.full(d, np.nan)
        self._m2 = np.full(d, np.nan)
        self._min = np.full(d, np.nan)
        self._max = np.full(d, np.nan)
        self._poisoned = np.zeros(d, dtype=bool)

    def _as_batch(self, batch):
        a = real_array(batch, allow_nan=self.nan_policy != "raise", nonempty=False)
        if a.ndim == 1:
            a = a[:, np.newaxis]
        elif a.ndim != 2:
            raise ValueError(f"batch must be 1-D or 2-D, got shape {a.shape}")
        a = a.astype(float, copy=False)
        if self.n_features is None:
            if a.shape[0] == 0:
                return a  # an empty first batch does not fix the feature count
            self._allocate(a.shape[1])
        elif a.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {a.shape[1]}")
        return a

    @staticmethod
    def _combine(na, ma, m2a, nb, mb, m2b):
        """Chan et al. combination of two (count, mean, M2) triples per column."""
        n = na + nb
        with np.errstate(invalid="ignore", divide="ignore"):
            delta = mb - ma
            mean = np.where(na == 0, mb, np.where(nb == 0, ma, ma + delta * nb / n))
            m2 = np.where(na == 0, m2b,
                          np.where(nb == 0, m2a, m2a + m2b + delta ** 2 * na * nb / n))
        return n, mean, m2

    # -- public API ---------------------------------------------------------- #

    def update(self, batch):
        """Fold a batch of observations into the accumulator and return self.

        Parameters
        ----------
        batch : array_like of real numbers, shape (n,) or (n, n_features)
            An empty batch (``n == 0``) is accepted and changes nothing.

        Raises
        ------
        TypeError
            Non-numeric input.
        ValueError
            Wrong dimensionality or feature count, infinities, NaN under
            ``'raise'``.
        """
        a = self._as_batch(batch)
        if a.shape[0] == 0:
            return self
        nan_mask = np.isnan(a)
        if self.nan_policy == "propagate":
            self._poisoned |= nan_mask.any(axis=0)
        nb = np.sum(~nan_mask, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mb = np.nanmean(a, axis=0)
            m2b = np.nansum((a - mb) ** 2, axis=0)
            m2b = np.where(nb == 0, np.nan, m2b)
            lo = np.nanmin(a, axis=0)
            hi = np.nanmax(a, axis=0)
        self._count, self._mean, self._m2 = self._combine(
            self._count, self._mean, self._m2, nb, mb, m2b)
        self._min = np.fmin(self._min, lo)
        self._max = np.fmax(self._max, hi)
        return self

    def merge(self, other):
        """Return a new accumulator equal to seeing both inputs' observations.

        Neither operand is modified. Merging with an accumulator that has not
        seen data returns a copy of the other. Both must share ``nan_policy``
        and, once set, ``n_features``.

        Raises
        ------
        TypeError
            ``other`` is not a RunningStats.
        ValueError
            Mismatched ``nan_policy`` or feature count.
        """
        if not isinstance(other, RunningStats):
            raise TypeError("can only merge with another RunningStats")
        if other.nan_policy != self.nan_policy:
            raise ValueError("cannot merge accumulators with different nan_policy")
        if self.n_features is None:
            return other.copy()
        if other.n_features is None:
            return self.copy()
        if self.n_features != other.n_features:
            raise ValueError(f"feature count mismatch: {self.n_features} != {other.n_features}")
        result = RunningStats(nan_policy=self.nan_policy)
        result._allocate(self.n_features)
        result._count, result._mean, result._m2 = self._combine(
            self._count, self._mean, self._m2, other._count, other._mean, other._m2)
        result._min = np.fmin(self._min, other._min)
        result._max = np.fmax(self._max, other._max)
        result._poisoned = self._poisoned | other._poisoned
        return result

    def copy(self):
        """Return an independent copy of this accumulator."""
        result = RunningStats(nan_policy=self.nan_policy)
        if self.n_features is not None:
            result.n_features = self.n_features
            for name in ("_count", "_mean", "_m2", "_min", "_max", "_poisoned"):
                setattr(result, name, getattr(self, name).copy())
        return result

    def _require_data(self):
        if self.n_features is None:
            raise RuntimeError("no data has been seen; call update first")

    def _masked(self, values):
        return np.where(self._poisoned, np.nan, values)

    @property
    def count(self):
        """Observations per column, shape (n_features,), int64."""
        self._require_data()
        return self._count.copy()

    @property
    def mean(self):
        """Per-column mean, shape (n_features,); NaN for unseen columns."""
        self._require_data()
        return self._masked(self._mean)

    def var(self, ddof=0):
        """Per-column variance with ``ddof``; NaN when ``count <= ddof``."""
        self._require_data()
        if isinstance(ddof, bool) or not isinstance(ddof, (int, np.integer)) or ddof < 0:
            raise ValueError("ddof must be a non-negative integer")
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(self._count > ddof, self._m2 / (self._count - ddof), np.nan)
        return self._masked(result)

    def std(self, ddof=0):
        """Per-column standard deviation: ``sqrt(var(ddof))``."""
        return np.sqrt(self.var(ddof))

    @property
    def min(self):
        """Per-column minimum, shape (n_features,)."""
        self._require_data()
        return self._masked(self._min)

    @property
    def max(self):
        """Per-column maximum, shape (n_features,)."""
        self._require_data()
        return self._masked(self._max)

    def __repr__(self):
        if self.n_features is None:
            return f"RunningStats(nan_policy={self.nan_policy!r}, empty)"
        return (f"RunningStats(nan_policy={self.nan_policy!r}, n_features={self.n_features}, "
                f"count={self._count.tolist()})")
