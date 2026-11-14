"""Add incremental fitting to the original sequential pipeline."""

import numpy as np

from numcompute.pipeline import Pipeline as _Pipeline, _transformed
from .metrics import _labels_array


class Pipeline(_Pipeline):
    """Update each transformer, transform the chunk, then update the model.

    X has shape (n,d), and an estimator requires matching (n,) labels.
    Each step must provide partial_fit. Predictions use the current transforms.
    Updating a scaler can therefore move features relative to a tree's learned
    thresholds until it retrains. Transformed feature width must stay fixed
    when the final estimator requires it, as the buffered tree does.
    """

    @property
    def n_buffered_(self):
        """Rows waiting in the final step's buffer (0 if it has none)."""
        return getattr(self.steps[-1][1], "n_buffered_", 0)

    @property
    def is_fitted_(self):
        """True once the final step can predict."""
        return self._is_fitted

    @property
    def named_steps(self):
        """Return the steps as a {name: step} dictionary."""
        return dict(self.steps)

    def _input(self, X):
        """Return X as a nonempty 2D array with the expected number of features."""
        X = np.asarray(X)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("X must be a nonempty 2D array")
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features")
        return X

    def fit(self, X, y=None):
        """Reset all steps on a complete dataset; return self."""
        return super().fit(np.asarray(X), y)

    def partial_fit(self, X, y=None):
        """Update from one chunk and return self; invalid shapes raise ValueError."""
        X = self._input(X)
        for name, step in self.steps:
            if not callable(getattr(step, "partial_fit", None)):
                raise TypeError(f"{name!r} must support partial_fit")
        model = self.steps[-1][1]
        predicts = callable(getattr(model, "predict", None))
        if predicts:
            y = _labels_array(y, "y")
            if len(y) != len(X):
                raise ValueError("X and y must have matching rows")
        values = X.copy()
        for name, step in self.steps[:-1]:
            step.partial_fit(values.copy())
            values = _transformed(step, name, values)
        if predicts:
            model.partial_fit(values.copy(), y.copy())
        else:
            model.partial_fit(values.copy())
        self._is_fitted = bool(getattr(model, "is_fitted_", True))
        self.n_features_in_ = X.shape[1]
        return self

    def _features(self, X):
        """Apply the current preprocessing steps to X for prediction."""
        if not self._is_fitted:
            raise RuntimeError("pipeline is not trained; call fit or fill the buffer")
        values = self._input(X).copy()
        for name, step in self.steps[:-1]:
            values = _transformed(step, name, values)
        return values

    def predict(self, X):
        """Return predictions after applying the current preprocessing steps."""
        return self.steps[-1][1].predict(self._features(X))

    def predict_proba(self, X):
        """Return (n,k) class probabilities."""
        return self.steps[-1][1].predict_proba(self._features(X))

    def transform(self, X):
        """Return features for a transformer-ending pipeline."""
        name, final = self.steps[-1]
        return _transformed(final, name, self._features(X))
