"""Composition contracts using small structural transformer/estimator doubles."""

import numpy as np
import pytest

from numcompute.pipeline import Estimator, FeatureUnion, Pipeline, Transformer


class AffineTransformer:
    """Accept any feature width and record the data used for fitting."""

    def __init__(self, scale=1, offset=0):
        self.scale = scale
        self.offset = offset
        self.fit_calls = 0

    def fit(self, X):
        self.fit_calls += 1
        self.fit_input = X.copy()
        return self

    def transform(self, X):
        return X * self.scale + self.offset

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class RecordingEstimator:
    def __init__(self):
        self.fit_calls = 0

    def fit(self, X, y=None):
        self.fit_calls += 1
        self.fit_input = X.copy()
        self.target = None if y is None else y.copy()
        return self

    def predict(self, X):
        if X.shape[1] != self.fit_input.shape[1]:
            raise ValueError("estimator feature count mismatch")
        return X.sum(axis=1)


class RowSum(AffineTransformer):
    def transform(self, X):
        return X.sum(axis=1, keepdims=True)


@pytest.fixture
def X():
    return np.array([[1., 2.], [3., 4.], [5., 6.]])


def test_structural_protocol_steps_and_sequential_fit_transform(X):
    # No Protocol inheritance is required; the annotations describe the API.
    first: Transformer = AffineTransformer(offset=3)
    final: Transformer = AffineTransformer(scale=2)
    pipe = Pipeline([("add", first), ("multiply", final)])

    assert pipe.fit(X, y=np.arange(3)) is pipe
    np.testing.assert_array_equal(first.fit_input, X)
    np.testing.assert_array_equal(final.fit_input, X + 3)
    result = pipe.transform(X)
    assert isinstance(result, np.ndarray)
    assert result.shape == X.shape
    np.testing.assert_array_equal(result, (X + 3) * 2)
    assert first.fit_calls == final.fit_calls == 1


@pytest.mark.parametrize("with_y", [False, True])
def test_estimator_ending_pipeline_forwards_y_and_predicts(X, with_y):
    transformer = AffineTransformer(offset=2)
    estimator: Estimator = RecordingEstimator()
    pipe = Pipeline([("offset", transformer), ("estimator", estimator)])
    y = np.array([0, 1, 0]) if with_y else None

    assert pipe.fit(X, y) is pipe
    np.testing.assert_array_equal(estimator.fit_input, X + 2)
    if with_y:
        np.testing.assert_array_equal(estimator.target, y)
    else:
        assert estimator.target is None
    result = pipe.predict(X)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(X),)
    np.testing.assert_array_equal(result, (X + 2).sum(axis=1))
    assert transformer.fit_calls == estimator.fit_calls == 1


@pytest.mark.parametrize("container", [Pipeline, FeatureUnion])
def test_fit_transform_fits_without_prior_fit_and_matches_separate_calls(X, container):
    def make():
        return container([("add", AffineTransformer(offset=1)),
                          ("scale", AffineTransformer(scale=3))])

    direct, separate = make(), make()
    result = direct.fit_transform(X)
    np.testing.assert_array_equal(result, separate.fit(X).transform(X))
    np.testing.assert_array_equal(result, direct.transform(X))


def test_feature_union_fits_same_input_and_concatenates_in_branch_order(X):
    first, second = RowSum(), AffineTransformer(scale=10)
    union = FeatureUnion([("sum", first), ("scale", second)])
    assert union.fit(X) is union
    for branch in (first, second):
        np.testing.assert_array_equal(branch.fit_input, X)
    result = union.transform(X)
    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result[:, :1], X.sum(axis=1, keepdims=True))
    np.testing.assert_array_equal(result[:, 1:], X * 10)
    assert first.fit_calls == second.fit_calls == 1


@pytest.mark.parametrize("container", [Pipeline, FeatureUnion])
def test_transform_accepts_different_input_feature_and_sample_counts(X, container):
    model = container([("scale", AffineTransformer(scale=2))]).fit(X)
    new_X = np.arange(20.).reshape(4, 5)
    result = model.transform(new_X)
    assert result.shape == new_X.shape
    np.testing.assert_array_equal(result, new_X * 2)


def test_estimator_validates_transformed_width_not_original_pipeline_width(X):
    estimator = RecordingEstimator()
    pipe = Pipeline([("sum", RowSum()), ("estimator", estimator)]).fit(X)
    new_X = np.arange(20.).reshape(4, 5)
    np.testing.assert_array_equal(pipe.predict(new_X), new_X.sum(axis=1))

    direct = Pipeline([("estimator", RecordingEstimator())]).fit(X)
    with pytest.raises(ValueError, match="estimator feature count mismatch"):
        direct.predict(new_X)


@pytest.mark.parametrize("kind, method", [
    ("transformer", "transform"), ("estimator", "predict"), ("union", "transform"),
])
def test_inference_before_fit_raises_clear_error(X, kind, method):
    if kind == "union":
        model = FeatureUnion([("branch", AffineTransformer())])
    else:
        step = RecordingEstimator() if kind == "estimator" else AffineTransformer()
        model = Pipeline([("step", step)])
    with pytest.raises(RuntimeError, match="call fit"):
        getattr(model, method)(X)


@pytest.mark.parametrize("container", [Pipeline, FeatureUnion])
@pytest.mark.parametrize("names, error, message", [
    ([], ValueError, "at least one"),
    ([""], ValueError, "non-empty"),
    (["  "], ValueError, "non-empty"),
    ([1], TypeError, "strings"),
    (["same", "same"], ValueError, "duplicate name"),
])
def test_invalid_names(container, names, error, message):
    with pytest.raises(error, match=message):
        container([(name, AffineTransformer()) for name in names])


@pytest.mark.parametrize("container", [Pipeline, FeatureUnion])
@pytest.mark.parametrize("entries", [None, ["bad"], [("name",)], [("a", object(), 3)]])
def test_malformed_entries(container, entries):
    with pytest.raises(TypeError, match="pair"):
        container(entries)


@pytest.mark.parametrize("position", ["intermediate", "final", "branch"])
@pytest.mark.parametrize("missing", ["fit", "transform", "fit_transform"])
def test_invalid_step_interfaces(position, missing):
    # Non-final steps need only fit and transform under the explicit contract.
    step = AffineTransformer()
    setattr(step, missing, None)
    if position == "intermediate" and missing == "fit_transform":
        Pipeline([("step", step), ("final", AffineTransformer())])
        return
    with pytest.raises(TypeError, match="callable|transformer or estimator"):
        if position == "branch":
            FeatureUnion([("step", step)])
        elif position == "intermediate":
            Pipeline([("step", step), ("final", AffineTransformer())])
        else:
            Pipeline([("step", step)])


def test_estimator_cannot_be_an_intermediate_step_or_union_branch():
    with pytest.raises(TypeError, match="transform"):
        Pipeline([("estimator", RecordingEstimator()), ("final", AffineTransformer())])
    with pytest.raises(TypeError, match="transform"):
        FeatureUnion([("estimator", RecordingEstimator())])


@pytest.mark.parametrize("method", ["transform", "fit_transform"])
def test_estimator_only_pipeline_rejects_transformation(X, method):
    estimator = RecordingEstimator()
    pipe = Pipeline([("estimator", estimator)]).fit(X)
    with pytest.raises(TypeError, match="transform"):
        getattr(pipe, method)(X)
    assert estimator.fit_calls == 1


def test_transformer_only_pipeline_rejects_prediction(X):
    pipe = Pipeline([("transformer", AffineTransformer())]).fit(X)
    with pytest.raises(TypeError, match="predict"):
        pipe.predict(X)


class DualStep(RecordingEstimator):
    def transform(self, X):
        return X + 1

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def test_dual_capability_final_receives_y_and_supports_both_operations(X):
    final = DualStep()
    y = np.array([1, 0, 1])
    pipe = Pipeline([("dual", final)])
    np.testing.assert_array_equal(pipe.fit_transform(X, y), X + 1)
    np.testing.assert_array_equal(final.target, y)
    np.testing.assert_array_equal(pipe.predict(X), X.sum(axis=1))
    assert final.fit_calls == 1


@pytest.mark.parametrize("container", [Pipeline, FeatureUnion])
@pytest.mark.parametrize("bad_X, error, message", [
    ([[1, 2]], TypeError, "NumPy array"),
    (np.array(1), ValueError, "two-dimensional"),
    (np.ones(3), ValueError, "two-dimensional"),
    (np.ones((2, 2, 2)), ValueError, "two-dimensional"),
])
@pytest.mark.parametrize("method", ["fit", "transform", "fit_transform"])
def test_invalid_input_arrays(X, container, bad_X, error, message, method):
    model = container([("step", AffineTransformer())]).fit(X)
    with pytest.raises(error, match=message):
        getattr(model, method)(bad_X)


@pytest.mark.parametrize("bad_X, error", [([1, 2], TypeError), (np.ones(3), ValueError)])
def test_predict_validates_input_array(X, bad_X, error):
    pipe = Pipeline([("estimator", RecordingEstimator())]).fit(X)
    with pytest.raises(error, match="NumPy array|two-dimensional"):
        pipe.predict(bad_X)


class InvalidOutput(AffineTransformer):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def transform(self, X):
        if self.mode == "rows":
            return X[:-1]
        if self.mode == "dimensions":
            return X[:, 0]
        return X.tolist()


@pytest.mark.parametrize("position", ["intermediate", "final", "branch"])
@pytest.mark.parametrize("mode, error, message", [
    ("rows", ValueError, "preserve the number of rows"),
    ("dimensions", ValueError, "two-dimensional"),
    ("list", TypeError, "NumPy array"),
])
def test_invalid_transformed_output(X, position, mode, error, message):
    bad = InvalidOutput(mode)
    with pytest.raises(error, match=message):
        if position == "intermediate":
            Pipeline([("bad", bad), ("final", AffineTransformer())]).fit(X)
        elif position == "final":
            Pipeline([("bad", bad)]).fit(X).transform(X)
        else:
            FeatureUnion([("bad", bad)]).fit(X).transform(X)


def test_predict_requires_numpy_output(X):
    class ListEstimator(RecordingEstimator):
        def predict(self, X):
            return [0] * len(X)

    pipe = Pipeline([("estimator", ListEstimator())]).fit(X)
    with pytest.raises(TypeError, match="predictions.*NumPy array"):
        pipe.predict(X)


class MutatingTransformer(AffineTransformer):
    def fit(self, X):
        super().fit(X)
        X[:] = -100
        return self

    def transform(self, X):
        X += 10
        return X


@pytest.mark.parametrize("container", [Pipeline, FeatureUnion])
def test_noncontiguous_input_preservation_and_copy_isolation(container):
    backing = np.arange(24.).reshape(4, 6)
    original = backing.copy()
    X = backing[:, ::2]
    first, second = MutatingTransformer(), AffineTransformer(scale=2)
    model = container([("mutating", first), ("recording", second)])
    result = model.fit_transform(X)

    np.testing.assert_array_equal(backing, original)
    assert backing.dtype == original.dtype
    np.testing.assert_array_equal(first.fit_input, X)
    if container is Pipeline:
        np.testing.assert_array_equal(second.fit_input, X + 10)
        np.testing.assert_array_equal(result, (X + 10) * 2)
    else:
        np.testing.assert_array_equal(second.fit_input, X)
        np.testing.assert_array_equal(result, np.concatenate([X + 10, X * 2], axis=1))


def test_estimator_cannot_mutate_caller_X_or_y(X):
    class MutatingEstimator(RecordingEstimator):
        def fit(self, X, y=None):
            super().fit(X, y)
            X[:] = -1
            y[:] = -1
            return self

        def predict(self, X):
            X[:] = 0
            return X.sum(axis=1)

    y = np.array([1, 2, 3])
    originals = (X.copy(), y.copy())
    pipe = Pipeline([("estimator", MutatingEstimator())]).fit(X, y)
    np.testing.assert_array_equal(pipe.predict(X), np.zeros(len(X)))
    np.testing.assert_array_equal(X, originals[0])
    np.testing.assert_array_equal(y, originals[1])


def test_nested_composition_uses_actual_final_capabilities(X):
    inner = Pipeline([("add", AffineTransformer(offset=1))])
    union = FeatureUnion([("inner", inner), ("sum", RowSum())])
    estimator = RecordingEstimator()
    pipe = Pipeline([("union", union), ("estimator", estimator)])
    y = np.array([0, 1, 0])
    pipe.fit(X, y)
    expected = np.concatenate([X + 1, X.sum(axis=1, keepdims=True)], axis=1)
    np.testing.assert_array_equal(estimator.fit_input, expected)
    np.testing.assert_array_equal(estimator.target, y)
    np.testing.assert_array_equal(pipe.predict(X), expected.sum(axis=1))


@pytest.mark.parametrize("container", [Pipeline, FeatureUnion])
def test_failed_refit_leaves_container_unfitted(X, container):
    class FailingTransformer(AffineTransformer):
        fail = False

        def fit(self, X):
            if self.fail:
                raise ValueError("step fit failed")
            return super().fit(X)

    step = FailingTransformer()
    model = container([("step", step)]).fit(X)
    step.fail = True
    with pytest.raises(ValueError, match="step fit failed"):
        model.fit(X)
    with pytest.raises(RuntimeError, match="call fit"):
        model.transform(X)
