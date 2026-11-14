"""Streaming preprocessing/statistics/metric invariants against batch oracles."""

import numpy as np
import pytest

from numcompute_stream import (
    StandardScaler,
    MinMaxScaler,
    SimpleImputer,
    OneHotEncoder,
    StreamingStats,
    StreamingMetrics,
)
from numcompute_stream import metrics as batch


@pytest.mark.parametrize("cls", [StandardScaler, MinMaxScaler])
def test_uneven_chunk_scalers_match_batch(cls):
    X = np.array([[1, np.nan, 2], [2, 4, 2], [7, 9, 2], [4, 3, 2.0]])
    incremental = cls().partial_fit(X[:1]).partial_fit(X[1:3]).partial_fit(X[3:])
    reference = cls().fit(X)
    np.testing.assert_allclose(incremental.transform(X), reference.transform(X), equal_nan=True)
    assert np.isfinite(incremental.transform([[1, 4, 2]])).all()


@pytest.mark.parametrize("cls", [StandardScaler, MinMaxScaler, SimpleImputer])
def test_initial_all_nan_can_recover_and_fit_resets(cls):
    obj = cls().partial_fit([[np.nan, 1.0]])
    assert np.isfinite(obj.transform([[2.0, 1.0]])).all()
    obj.partial_fit([[4.0, 3.0]])
    assert np.isfinite(obj.transform([[np.nan, 1.0]])[:, 1]).all()
    obj.fit([[5.0, 6.0], [7.0, 8.0]])
    expected = cls().fit([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_allclose(obj.transform([[5.0, 6.0]]), expected.transform([[5.0, 6.0]]))


@pytest.mark.parametrize("cls", [StandardScaler, MinMaxScaler, SimpleImputer])
def test_preprocessor_shape_infinite_and_unfitted_errors(cls):
    obj = cls()
    with pytest.raises(RuntimeError):
        obj.transform([[1.0]])
    obj.partial_fit([[1.0]])
    with pytest.raises(ValueError):
        obj.partial_fit([[1.0, 2.0]])
    with pytest.raises(ValueError):
        obj.partial_fit([[np.inf]])
    with pytest.raises(ValueError):
        obj.partial_fit(np.empty((0, 1)))


@pytest.mark.parametrize("strategy", ["mean", "median", "constant"])
def test_imputer_stream_matches_combined_data(strategy):
    X = np.array([[1.0, np.nan], [5.0, 2.0], [np.nan, 7.0]])
    obj = SimpleImputer(strategy, fill_value=-1).partial_fit(X[:1]).partial_fit(X[1:])
    expected = SimpleImputer(strategy, fill_value=-1).fit(X)
    np.testing.assert_allclose(obj.statistics_, expected.statistics_)
    np.testing.assert_allclose(obj.transform(X), expected.transform(X))


def test_welford_large_offset_and_copy_isolation():
    X = 1e10 + np.random.default_rng(6).normal(size=(101, 2))
    obj = StandardScaler().partial_fit(X[:37]).partial_fit(X[37:])
    np.testing.assert_allclose(obj.mean_, X.mean(axis=0), rtol=1e-14)
    np.testing.assert_allclose(obj.var_, X.var(axis=0), rtol=2e-6)
    original = X.copy()
    obj.transform(X[:, ::-1])
    np.testing.assert_array_equal(X, original)


def test_encoder_appends_without_moving_existing_columns():
    encoder = OneHotEncoder(handle_unknown="ignore").partial_fit([["b", "z"], ["c", "z"]])
    original = encoder.transform([["b", "z"]])
    encoder.partial_fit([["a", "y"]])
    np.testing.assert_array_equal(encoder.transform([["b", "z"]])[:, :3], original)
    assert encoder.n_features_out_ == 5
    np.testing.assert_array_equal(encoder.transform([["unknown", "unknown"]]), np.zeros((1, 5)))
    encoder.fit([["fresh", "start"]])
    assert encoder.n_features_out_ == 2


def test_encoder_numeric_dtype_changes_and_unknowns():
    enc = OneHotEncoder().partial_fit([[1], [2]])
    with pytest.raises(ValueError):
        enc.transform([[3]])
    enc.partial_fit(np.array([[3.0]]))
    assert enc.transform([[3]]).sum() == 1
    with pytest.raises(TypeError):
        enc.partial_fit([["text"]])


@pytest.mark.parametrize("window", [None, 4])
def test_statistics_match_batch_with_window(window):
    X = np.arange(18.0).reshape(9, 2)
    X[3, 1] = np.nan
    obj = StreamingStats(bins=[0, 5, 10, 20], window_size=window)
    obj.update_stats(X[:2]).update_stats(X[2:7]).update_stats(X[7:])
    reference = X if window is None else X[-window:]
    result = obj.result()
    np.testing.assert_allclose(result["mean"], np.nanmean(reference, axis=0))
    np.testing.assert_allclose(result["variance"], np.nanvar(reference, axis=0))
    np.testing.assert_allclose(
        result["quantiles"], np.nanquantile(reference, [0.25, 0.5, 0.75], axis=0)
    )
    for col in range(2):
        np.testing.assert_array_equal(
            result["histogram"][col],
            np.histogram(reference[:, col][np.isfinite(reference[:, col])], bins=obj.bins)[0],
        )
    result["mean"][:] = 999
    assert np.all(obj.result()["mean"] < 999)
    obj.reset()
    with pytest.raises(RuntimeError):
        obj.result()


@pytest.mark.parametrize(
    "kwargs", [{"bins": [1, 1]}, {"bins": [0, np.inf]}, {"quantiles": [2]}, {"window_size": 0}]
)
def test_invalid_stream_stats_configuration(kwargs):
    with pytest.raises(ValueError):
        StreamingStats(**kwargs)


@pytest.mark.parametrize("average", ["binary", "micro", "macro", "weighted", None])
@pytest.mark.parametrize("window", [None, 5])
def test_stream_metrics_match_batch(average, window):
    t = np.array([0, 1, 0, 1, 1, 0, 1, 1])
    p = np.array([0, 0, 1, 1, 1, 0, 0, 1])
    scores = np.array([0.1, 0.4, 0.8, 0.7, 0.6, 0.2, 0.3, 0.9])
    obj = StreamingMetrics(labels=[0, 1], average=average, window_size=window, track_auc=True)
    obj.update(t[:3], p[:3], scores[:3]).update(t[3:6], p[3:6], scores[3:6]).update(
        t[6:], p[6:], scores[6:]
    )
    result = obj.result()
    tail = slice(None) if window is None else slice(-window, None)
    tt, pp, ss = t[tail], p[tail], scores[tail]
    assert result["accuracy"] == batch.accuracy(tt, pp)
    for name in ("precision", "recall", "f1"):
        np.testing.assert_allclose(
            result[name], getattr(batch, name)(tt, pp, average=average, labels=[0, 1])
        )
    np.testing.assert_array_equal(
        result["confusion_matrix"], batch.confusion_matrix(tt, pp, labels=[0, 1])
    )
    assert result["roc_auc"] == batch.roc_auc_score(tt, ss)
    assert result["n_samples"] == len(tt)


def test_metrics_new_labels_order_and_reset():
    obj = StreamingMetrics().update(["z", "a"], ["z", "a"]).update(["b"], ["z"])
    np.testing.assert_array_equal(obj.result()["labels"], ["a", "b", "z"])
    assert obj.result()["accuracy"] == 2 / 3
    result = obj.result()
    result["confusion_matrix"][:] = 0
    assert obj.result()["confusion_matrix"].sum() == 3
    obj.reset()
    assert obj.result()["n_samples"] == 0 and np.isnan(obj.result()["accuracy"])


def test_rolling_metrics_evict_labels_and_bound_history():
    obj = StreamingMetrics(window_size=3)
    for i in range(20):
        obj.update([i], [i])
    np.testing.assert_array_equal(obj.result()["labels"], [17, 18, 19])
    assert len(obj._counts) == 3 and len(obj._history) == 3


def test_metrics_auc_single_class_then_second_and_ties():
    obj = StreamingMetrics(track_auc=True).update([0, 0], [0, 0], [0.5, 0.5])
    assert np.isnan(obj.result()["roc_auc"])
    obj.update([1, 1], [1, 1], [0.5, 0.5])
    assert obj.result()["roc_auc"] == 0.5


@pytest.mark.parametrize("t,p,s", [([0], [0, 1], [0.0]), ([0], [0], [np.nan]), ([0], [0], None)])
def test_metrics_invalid_update_does_not_change_counts(t, p, s):
    obj = StreamingMetrics(track_auc=True)
    with pytest.raises(ValueError):
        obj.update(t, p, s)
    assert obj.result()["n_samples"] == 0


def test_mixed_signed_unsigned_labels_remain_exact_in_streaming_counts():
    t = np.array([-1, 2], dtype=np.int64)
    p = np.array([2, 2**64 - 1], dtype=np.uint64)
    obj = StreamingMetrics().update(t, p)
    result = obj.result()
    assert result["labels"].tolist() == [-1, 2, 2**64 - 1]
    np.testing.assert_array_equal(result["confusion_matrix"], [[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    obj = StreamingMetrics(window_size=1).update(t, p)
    assert obj.result()["labels"].tolist() == [2, 2**64 - 1]
