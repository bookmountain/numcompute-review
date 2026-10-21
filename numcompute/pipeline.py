"""Minimal sequential and parallel feature composition using NumPy.

Feature arrays have shape (n_samples, n_features). Dtype and value policies
belong to individual steps, allowing categorical data and missing values.
Objects are fitted in place; arrays passed to their methods are copied to
protect caller-owned storage. Object-array copies are shallow, so steps must
not mutate Python objects stored inside array elements.
"""

from typing import Protocol, TypeVar

import numpy as np


_TransformerT = TypeVar("_TransformerT", bound="Transformer")
_EstimatorT = TypeVar("_EstimatorT", bound="Estimator")


class Transformer(Protocol):
    """Structural interface for fitted, row-preserving feature transforms.

    X and transformed outputs are two-dimensional NumPy arrays. Implementations
    document dtype restrictions, errors, and their own time/space complexity.
    """

    def fit(self: _TransformerT, X: np.ndarray) -> _TransformerT:
        """Learn from X without modifying it and return self."""
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return transformed features without re-fitting; reject unfitted use."""
        ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Return the equivalent of fit(X).transform(X), preserving X."""
        ...


class Estimator(Protocol):
    """Structural interface for fitted predictors; no model is supplied here.

    X has shape (n_samples, n_features); y and prediction shapes are defined by
    the estimator. Implementations validate feature counts and document their
    errors and time/space complexity.
    """

    def fit(self: _EstimatorT, X: np.ndarray, y=None) -> _EstimatorT:
        """Learn from X and optional y without modifying them; return self."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return NumPy predictions using fitted state; reject unfitted use."""
        ...


def _named_objects(items, label):
    try:
        entries = tuple(items)
    except TypeError as exc:
        raise TypeError(f"{label} must be an iterable of (name, object) pairs") from exc
    if not entries:
        raise ValueError(f"{label} must contain at least one entry")
    names = set()
    result = []
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise TypeError(f"each {label} entry must be a (name, object) pair")
        name, obj = entry
        if not isinstance(name, str):
            raise TypeError("names must be strings")
        if not name.strip():
            raise ValueError("names must be non-empty")
        if name in names:
            raise ValueError(f"duplicate name: {name!r}")
        names.add(name)
        result.append((name, obj))
    return tuple(result)


def _supports(obj, method):
    # A nested pipeline exposes methods whose availability depends on its end.
    if isinstance(obj, Pipeline) and method in ("transform", "fit_transform", "predict"):
        operation = "transform" if method == "fit_transform" else method
        return _supports(obj.steps[-1][1], operation)
    return callable(getattr(obj, method, None))


def _require(obj, name, methods):
    for method in methods:
        if not _supports(obj, method):
            raise TypeError(f"{name!r} must provide callable {method}()")


def _matrix(X, label="X"):
    if not isinstance(X, np.ndarray):
        raise TypeError(f"{label} must be a NumPy array")
    if X.ndim != 2:
        raise ValueError(f"{label} must be two-dimensional")
    return X


def _transformed(obj, name, X):
    result = _matrix(obj.transform(X.copy()), f"output of {name!r}")
    if result.shape[0] != X.shape[0]:
        raise ValueError(f"{name!r} must preserve the number of rows")
    return result


def _fitted_input(obj, X):
    if not obj._is_fitted:
        raise RuntimeError("call fit() before transform() or predict()")
    return _matrix(X)


class Pipeline:
    """Fit and apply an ordered sequence of named steps.

    Parameters
    ----------
    steps : iterable of (str, object) pairs
        Nonempty, uniquely named steps. Non-final steps provide fit/transform;
        the final step follows the Transformer or Estimator interface.
        Supplied objects are fitted directly, never cloned. If the final step
        supports both transform and predict, it is fitted as an estimator.

    Notes
    -----
    X must be a two-dimensional NumPy array. Transforms preserve rows and may
    change feature counts. Input feature counts may differ between fitting and
    inference if the individual steps accept them. The final estimator validates
    the feature count it receives after preceding transformations.
    n_features_in_ records the fitted input width as metadata, not a restriction.
    Empty dimensions and dtype/value restrictions are delegated to steps.
    Construction takes O(s) time/space for s steps. Execution costs the sum of
    invoked step costs plus O(sum N_i) for protective array copies, where N_i
    are the sizes passed to methods. Extra live array storage is O(max N_i),
    excluding step internals, retained fitted data, and returned outputs.

    Raises
    ------
    TypeError
        For malformed entries or missing callable methods.
    ValueError
        For empty steps or empty/duplicate names.
    """

    def __init__(self, steps):
        self.steps = _named_objects(steps, "steps")
        for name, step in self.steps[:-1]:
            _require(step, name, ("fit", "transform"))
        name, final = self.steps[-1]
        _require(final, name, ("fit",))
        if not _supports(final, "predict"):
            if not _supports(final, "transform"):
                raise TypeError(f"{name!r} must be a transformer or estimator")
            _require(final, name, ("fit_transform",))
        self._is_fitted = False

    def fit(self, X: np.ndarray, y=None) -> "Pipeline":
        """Fit in order and return self, forwarding y only to a final estimator.

        X has shape (n_samples, n_features); y follows the final estimator's
        contract. NumPy y arrays are copied before forwarding. Invalid input or
        intermediate output types/shapes raise TypeError/ValueError; step errors
        propagate. A failed fit leaves this pipeline unfitted. Time/space costs
        follow the class notes, plus O(y.size) if y is a NumPy array.
        """
        self._is_fitted = False
        values = _matrix(X)
        for name, step in self.steps[:-1]:
            step.fit(values.copy())
            values = _transformed(step, name, values)
        final = self.steps[-1][1]
        if _supports(final, "predict"):
            final.fit(values.copy(), y.copy() if isinstance(y, np.ndarray) else y)
        else:
            final.fit(values.copy())
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return row-preserving 2-D features using fitted steps, without fitting.

        Raises RuntimeError before fit, TypeError for unsupported transformation
        or non-array input/output, and ValueError for invalid dimensions or
        changed row count. Feature-count restrictions and their errors belong
        to individual steps. Costs follow the class notes.
        """
        values = _fitted_input(self, X)
        _require(self.steps[-1][1], self.steps[-1][0], ("transform",))
        for name, step in self.steps:
            values = _transformed(step, name, values)
        return values

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Return fit(X, y).transform(X) for a transformer-ending pipeline.

        Rejects unsupported transformation with TypeError before fitting.
        Shapes/errors are those of fit and transform; costs are their sum.
        """
        _require(self.steps[-1][1], self.steps[-1][0], ("transform",))
        return self.fit(X, y).transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Transform X through preceding steps and return final predictions.

        Prediction shape is defined by the estimator. Raises RuntimeError before
        fit, TypeError for unsupported prediction or non-array input/output, and
        ValueError for invalid feature-array shapes. Feature-count validation
        is delegated to individual steps, including the final estimator on its
        transformed input. Step errors propagate. No step is re-fitted.
        Costs follow the class notes, including the final estimator's prediction.
        """
        values = _fitted_input(self, X)
        name, final = self.steps[-1]
        _require(final, name, ("predict",))
        for step_name, step in self.steps[:-1]:
            values = _transformed(step, step_name, values)
        result = final.predict(values.copy())
        if not isinstance(result, np.ndarray):
            raise TypeError(f"predictions from {name!r} must be a NumPy array")
        return result


class FeatureUnion:
    """Combine named transformer outputs column-wise in branch order.

    Parameters
    ----------
    branches : iterable of (str, Transformer) pairs
        At least one uniquely named branch implementing the full transformer
        interface. Objects are fitted directly. Each branch receives an
        independent array copy containing the same original X values.

    Notes
    -----
    Input shape is (n_samples, n_features). Each branch returns a 2-D array with
    n_samples rows. The result has sum(branch_output_features) columns; NumPy
    determines the concatenated dtype. Input feature counts may differ between
    fit and transform unless an individual branch rejects them. n_features_in_
    records the fitted input width as metadata, not a restriction.
    Construction takes O(b) time/space for b
    branches. Fit costs branch fitting plus O(b*N) copying time and O(N) extra
    live array storage. Transform adds O(M) concatenation time/storage for M
    output elements, plus branch costs and O(b*N) input copying time. These
    bounds exclude branch internals and retained fitted data.

    Raises
    ------
    TypeError
        For malformed branches or missing callable methods.
    ValueError
        For empty branches or empty/duplicate names.
    """

    def __init__(self, branches):
        self.branches = _named_objects(branches, "branches")
        for name, branch in self.branches:
            _require(branch, name, ("fit", "transform", "fit_transform"))
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "FeatureUnion":
        """Fit every branch on the same X values and return self.

        X must be a 2-D NumPy array; otherwise raise TypeError/ValueError.
        Branch errors propagate and a failed fit leaves this union unfitted.
        Fitting costs and input-copy storage are described in the class notes.
        """
        self._is_fitted = False
        X = _matrix(X)
        for _, branch in self.branches:
            branch.fit(X.copy())
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return fitted branch outputs concatenated along the feature axis.

        Raises RuntimeError before fit, TypeError for non-array input/output,
        and ValueError for invalid dimensions or changed branch row count.
        Branch-specific feature-count errors propagate. Shape and costs follow
        the class notes.
        """
        X = _fitted_input(self, X)
        outputs = [_transformed(branch, name, X) for name, branch in self.branches]
        return np.concatenate(outputs, axis=1)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Return fit(X).transform(X); shapes/errors and costs combine both calls."""
        return self.fit(X).transform(X)
