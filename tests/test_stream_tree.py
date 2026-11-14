"""Decision-tree correctness and streaming state regression checks."""

import numpy as np
import pytest

from numcompute_stream import DecisionTreeClassifier


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_numeric_multiclass_and_probabilities(criterion):
    X = np.arange(18).reshape(9, 2)
    y = np.repeat(["a", "b", "c"], 3)
    tree = DecisionTreeClassifier(criterion=criterion).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), y)
    np.testing.assert_allclose(tree.predict_proba(X).sum(axis=1), 1)
    assert tree.score(X, y) == 1


def test_string_features_rejected_with_encoding_hint():
    with pytest.raises(TypeError, match="one-hot encode"):
        DecisionTreeClassifier().fit([["low"], ["high"]], [0, 1])


def test_depth_zero_and_majority_tie():
    tree = DecisionTreeClassifier(max_depth=0).fit([[0], [1]], ["z", "a"])
    np.testing.assert_array_equal(tree.predict([[0], [100]]), ["a", "a"])
    np.testing.assert_array_equal(tree.predict_proba([[1]]), [[0.5, 0.5]])


@pytest.mark.parametrize("kwargs", [{"min_samples_split": 5}, {"max_depth": 0}])
def test_stopping_limits(kwargs):
    tree = DecisionTreeClassifier(**kwargs).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert tree.root_.feature == -1


def test_constant_features_cannot_split():
    tree = DecisionTreeClassifier().fit(np.ones((4, 2)), [0, 1, 1, 1])
    assert tree.root_.feature == -1
    np.testing.assert_array_equal(tree.predict([[1, 1]]), [1])


def test_partial_buffer_threshold_overshoot_and_clear():
    tree = DecisionTreeClassifier(buffer_size=3)
    tree.partial_fit([[0], [1]], [0, 0])
    assert not tree.is_fitted_ and tree.n_buffered_ == 2
    with pytest.raises(RuntimeError):
        tree.predict([[0]])
    tree.partial_fit([[2], [3]], [1, 1])
    assert tree.n_retrains_ == 1 and tree.n_buffered_ == 0
    np.testing.assert_array_equal(tree.predict([[0], [3]]), [0, 1])
    tree.partial_fit([[100]], [9])
    np.testing.assert_array_equal(tree.predict([[0], [3]]), [0, 1])
    tree.partial_fit([[101], [102]], [9, 9])
    assert tree.n_retrains_ == 2
    np.testing.assert_array_equal(tree.classes_, [9])
    assert tree.predict([[0]])[0] == 9


def test_buffer_owns_arrays_and_fit_resets():
    X, y = np.array([[0.0], [1.0]]), np.array([0, 1])
    tree = DecisionTreeClassifier(buffer_size=4).partial_fit(X, y)
    X[:] = 100
    y[:] = 8
    tree.partial_fit([[2], [3]], [1, 1])
    assert tree.predict([[0]])[0] == 0
    tree.partial_fit([[3]], [1])
    tree.fit([[0], [1]], [2, 2])
    assert tree.n_buffered_ == tree.n_retrains_ == 0
    np.testing.assert_array_equal(tree.classes_, [2])


@pytest.mark.parametrize("max_features", [1, "sqrt", "log2"])
def test_feature_subsampling_reproducible(max_features):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 5))
    y = (X[:, 2] > 0).astype(int)
    a = DecisionTreeClassifier(max_features=max_features, random_state=4).fit(X, y)
    b = DecisionTreeClassifier(max_features=max_features, random_state=4).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"max_depth": -1}, ValueError),
        ({"max_depth": True}, TypeError),
        ({"buffer_size": 0}, ValueError),
        ({"buffer_size": 1.5}, TypeError),
        ({"min_samples_split": 1}, ValueError),
        ({"max_features": "bad"}, ValueError),
        ({"criterion": "bad"}, ValueError),
        ({"max_features": False}, TypeError),
    ],
)
def test_invalid_parameters(kwargs, error):
    with pytest.raises(error):
        DecisionTreeClassifier(**kwargs)


@pytest.mark.parametrize(
    "X,y,error",
    [
        ([], [], ValueError),
        ([[1]], [], ValueError),
        ([[np.nan]], [0], ValueError),
        ([[np.inf]], [0], ValueError),
        ([[1], [2]], [0], ValueError),
        ([[1]], [np.nan], ValueError),
        (np.array([[object()]]), [0], TypeError),
    ],
)
def test_bad_training_data(X, y, error):
    with pytest.raises(error):
        DecisionTreeClassifier().fit(X, y)


def test_noncontiguous_input_preserved_and_width_checked():
    data = np.arange(24.0).reshape(6, 4)
    original = data.copy()
    tree = DecisionTreeClassifier().fit(data[:, ::2], [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(data, original)
    with pytest.raises(ValueError):
        tree.predict([[1]])
    with pytest.raises(TypeError):
        tree.predict([["a", "b"]])
    assert tree.predict(np.empty((0, 2))).shape == (0,)


def test_extreme_integer_thresholds_do_not_overflow_or_round():
    X = np.array([[2**63 + 1], [2**63 + 2], [2**64 - 1]], dtype=np.uint64)
    tree = DecisionTreeClassifier().fit(X, [0, 1, 1])
    np.testing.assert_array_equal(tree.predict(X), [0, 1, 1])


def test_stream_rejects_schema_change_without_appending():
    tree = DecisionTreeClassifier(buffer_size=4).partial_fit([[1]], [0])
    with pytest.raises(ValueError):
        tree.partial_fit([[1, 2]], [0])
    with pytest.raises(TypeError):
        tree.partial_fit([["x"]], [0])
    assert tree.n_buffered_ == 1
