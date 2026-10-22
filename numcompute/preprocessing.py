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
