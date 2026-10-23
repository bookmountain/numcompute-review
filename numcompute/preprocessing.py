"""NumPy tools for scaling, imputing and encoding feature data."""

import numpy as np

from ._validation import matrix, fitted


class StandardScaler:
    """Scale each feature using its training mean and standard deviation.

    NaN values are ignored when fitting and preserved when transforming.
    Training columns containing only NaN are rejected.
    Constant training columns use a scale of one.

    Attributes
    ----------
    mean_ : numpy.ndarray
        Mean of each training feature.
    scale_ : numpy.ndarray
        Standard deviation of each training feature, with zeros
        replaced by one.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def fit(self, X):
        """Learn column statistics from a 2D training array.

        Returns
        -------
        StandardScaler
            This fitted scaler.
        """
        X = matrix(X, allow_nan=True)

        if np.isnan(X).all(axis=0).any():
            raise ValueError(
                "cannot fit a feature containing only NaN values"
            )

        mean = np.nanmean(X, axis=0)
        scale = np.nanstd(X, axis=0)

        self.mean_ = mean
        self.scale_ = np.where(scale == 0, 1.0, scale)
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        """Scale a 2D array using the saved training statistics.

        Return a new array without changing the input.
        """
        X = fitted(self, X, allow_nan=True)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """Learn training statistics and scale the same array."""
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Scale features using their training minimum and maximum.

    NaN values are ignored when fitting and preserved when transforming.
    Training columns containing only NaN are rejected.
    Constant training columns use a range of one.
    New values outside the training range may fall outside [0, 1].

    Attributes
    ----------
    data_min_ : numpy.ndarray
        Minimum of each training feature.
    data_max_ : numpy.ndarray
        Maximum of each training feature.
    data_range_ : numpy.ndarray
        Difference between each training maximum and minimum.
    n_features_in_ : int
        Number of features seen during fitting.
    """

    def fit(self, X):
        """Learn feature limits from a 2D training array.

        Returns
        -------
        MinMaxScaler
            This fitted scaler.
        """
        X = matrix(X, allow_nan=True)

        if np.isnan(X).all(axis=0).any():
            raise ValueError(
                "cannot fit a feature containing only NaN values"
            )

        data_min = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)

        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_max - data_min
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        """Scale a 2D array using the saved training limits.

        Return a new array without changing the input.
        """
        X = fitted(self, X, allow_nan=True)
        denominator = np.where(
            self.data_range_ == 0, 1.0, self.data_range_
        )
        return (X - self.data_min_) / denominator

    def fit_transform(self, X):
        """Learn training limits and scale the same array."""
        return self.fit(X).transform(X)


class SimpleImputer:
    """Replace missing numeric values using training statistics.

    Parameters
    ----------
    strategy : {"mean", "median", "constant"}, default="mean"
        Method used to choose replacement values.
    fill_value : number, default=0.0
        Finite replacement used when strategy="constant".

    Attributes
    ----------
    statistics_ : numpy.ndarray
        Replacement value for each feature.
    n_features_in_ : int
        Number of features seen during fitting.

    Notes
    -----
    Missing values are represented by NaN.
    Mean and median strategies reject training columns containing
    only NaN. The constant strategy can fill those columns.
    """

    def __init__(self, strategy="mean", fill_value=0.0):
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X):
        """Learn replacement values from a 2D training array.

        Returns
        -------
        SimpleImputer
            This fitted imputer.
        """
        if self.strategy not in ("mean", "median", "constant"):
            raise ValueError(
                "strategy must be 'mean', 'median', or 'constant'"
            )

        X = matrix(X, allow_nan=True)

        if self.strategy == "constant":
            value = np.asarray(self.fill_value)

            if value.ndim != 0 or value.dtype.kind not in "biuf":
                raise TypeError("fill_value must be a real number")

            if not np.isfinite(value):
                raise ValueError("fill_value must be finite")

            statistics = np.full(X.shape[1], float(value))
        else:
            if np.isnan(X).all(axis=0).any():
                raise ValueError(
                    "cannot learn a replacement for a feature "
                    "containing only NaN values"
                )

            if self.strategy == "mean":
                statistics = np.nanmean(X, axis=0)
            else:
                statistics = np.nanmedian(X, axis=0)

        self.statistics_ = statistics
        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X):
        """Replace NaN using the saved training replacement values.

        Return a new array without changing the input.
        """
        X = fitted(self, X, allow_nan=True)
        return np.where(np.isnan(X), self.statistics_, X)

    def fit_transform(self, X):
        """Learn replacement values and fill the same array."""
        return self.fit(X).transform(X)


class OneHotEncoder:
    """Encode categories as a dense array of zeros and ones.

    Parameters
    ----------
    handle_unknown : {"error", "ignore"}, default="error"
        Raise an error for unseen categories, or encode them as
        zeros within the corresponding feature's output columns.

    Attributes
    ----------
    categories_ : list of numpy.ndarray
        Sorted training categories for each input feature.
    n_features_in_ : int
        Number of input features seen during fitting.
    n_features_out_ : int
        Number of encoded output columns.

    Notes
    -----
    Input must be a 2D array of strings or finite real numbers.
    Object arrays are not supported. Use a string array for text
    categories and a numeric array for numeric categories.
    Output columns are grouped by input feature.
    """

    def __init__(self, handle_unknown="error"):
        self.handle_unknown = handle_unknown

    def _check_input(self, X, nonempty=True):
        """Check categorical input without converting its values."""
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("expected a 2D categorical array")

        if X.shape[1] == 0:
            raise ValueError("at least one feature is required")

        if nonempty and X.shape[0] == 0:
            raise ValueError("training array must not be empty")

        if X.dtype.kind not in "biufUS":
            raise TypeError(
                "expected strings or real numeric categories"
            )

        if X.dtype.kind in "biuf" and not np.isfinite(X).all():
            raise ValueError("numeric categories must be finite")

        return X

    def fit(self, X):
        """Learn the distinct categories of each training feature."""
        if self.handle_unknown not in ("error", "ignore"):
            raise ValueError(
                "handle_unknown must be 'error' or 'ignore'"
            )

        X = self._check_input(X)

        pairs = np.empty(
            X.size,
            dtype=[("feature", np.intp), ("value", X.dtype)],
        )
        pairs["feature"] = np.tile(
            np.arange(X.shape[1]), X.shape[0]
        )
        pairs["value"] = X.ravel()

        unique_pairs = np.unique(pairs)
        feature_indices = unique_pairs["feature"]
        categories = unique_pairs["value"]
        counts = np.bincount(
            feature_indices, minlength=X.shape[1]
        )

        self.categories_ = np.split(
            categories.copy(), np.cumsum(counts)[:-1]
        )
        self._feature_indices = feature_indices.copy()
        self._categories = categories.copy()
        self._category_starts = np.r_[0, np.cumsum(counts)[:-1]]
        self._input_kind = X.dtype.kind
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = categories.size

        return self

    def transform(self, X):
        """Encode data using the saved training categories.

        Return a new floating-point array without changing the input.
        """
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("call fit before transform")

        X = self._check_input(X, nonempty=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, "
                f"got {X.shape[1]}"
            )

        if X.dtype.kind != self._input_kind:
            raise TypeError(
                "category dtype kind must match the training data"
            )

        matches = (
            X[:, self._feature_indices] == self._categories
        )

        known = np.logical_or.reduceat(
            matches, self._category_starts, axis=1
        )

        if self.handle_unknown == "error" and not known.all():
            raise ValueError("input contains unknown categories")

        return matches.astype(float)

    def fit_transform(self, X):
        """Learn training categories and encode the same array."""
        return self.fit(X).transform(X)
