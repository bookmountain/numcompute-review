"""Regression checks for exact labels/scores, variance and small workloads."""
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from benchmark.run import available_cases
from numcompute import metrics, stats


@pytest.mark.parametrize('reverse', [False, True])
def test_mixed_integer_labels_remain_distinct(reverse):
    high = np.array([2**63, 2**63 + 1], dtype=np.uint64)
    low = np.array([-1, -1], dtype=np.int64)
    t, p = (low, high) if reverse else (high, low)
    before = (t.copy(), p.copy())
    expected = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    assert_array_equal(metrics.confusion_matrix(t, p), expected.T if reverse else expected)
    assert_array_equal(t, before[0])
    assert_array_equal(p, before[1])


def test_explicit_unsigned_labels_preserve_order_and_membership():
    labels = np.array([2**63 + 1, 0, 2**63], dtype=np.uint64)
    t = np.array([2**63, 2**63 + 1], dtype=np.uint64)
    p = np.array([0, 0], dtype=np.int64)
    assert_array_equal(metrics.confusion_matrix(t, p, labels), [[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match='not in labels'):
        metrics.confusion_matrix(t, p, labels[:2])


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_explicit_labels_must_be_finite(bad):
    with pytest.raises(ValueError, match='NaN or infinite'):
        metrics.confusion_matrix([0, 1], [0, 1], labels=[0, 1, bad])


@pytest.mark.parametrize('scores', [
    np.array([2**63, 2**63 + 1], dtype=np.uint64),
    np.array([-(2**63), 2**63 - 1], dtype=np.int64),
    np.array([0, 2**64 - 1], dtype=np.uint64),
    np.array([False, True]),
])
def test_integer_roc_preserves_order_and_exact_thresholds(scores):
    before = scores.copy()
    assert metrics.roc_auc_score([0, 1], scores) == 1.0
    assert metrics.roc_auc_score([1, 0], scores) == 0.0
    fpr, tpr, thresholds = metrics.roc_curve([0, 1], scores)
    assert_array_equal(fpr, [0, 0, 1])
    assert_array_equal(tpr, [0, 1, 1])
    assert thresholds.tolist() == [np.inf, int(scores[1]), int(scores[0])]
    assert_array_equal(scores, before)


def test_large_integer_score_ties_match_pairwise_auc():
    scores = np.array([2**63, 2**63 + 1, 2**63 + 1, 2**63 + 2], dtype=np.uint64)
    assert metrics.roc_auc_score([0, 1, 0, 1], scores) == 0.875


@pytest.mark.parametrize('policy', ['propagate', 'raise', 'omit'])
@pytest.mark.parametrize('ddof', [2, 3])
@pytest.mark.parametrize('function', [stats.var, stats.std])
def test_exhausted_degrees_of_freedom_warn_and_return_nan(policy, ddof, function):
    with pytest.warns(RuntimeWarning, match='ddof'):
        result = function([1., 2.], ddof=ddof, nan_policy=policy)
    assert np.isscalar(result) and np.isnan(result)


def test_variance_omission_counts_per_slice_with_keepdims():
    with pytest.warns(RuntimeWarning, match='ddof'):
        result = stats.var([[1., 1.], [np.nan, 3.]], axis=0, ddof=1,
                           keepdims=True, nan_policy='omit')
    assert_allclose(result, [[np.nan, 2.]])
    with pytest.warns(RuntimeWarning, match='ddof'):
        result = stats.var(np.arange(8.).reshape(2, 2, 2), axis=(0, 2), ddof=4, keepdims=True)
    assert result.shape == (1, 2, 1) and np.isnan(result).all()


@pytest.mark.parametrize('policy', ['propagate', 'omit', 'raise'])
def test_mean_float32_honors_documented_float64_result(policy):
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert stats.mean(x, nan_policy=policy).dtype == np.float64
    assert stats.mean(x, axis=0, nan_policy=policy).dtype == np.float64


@pytest.mark.parametrize('rows', [0, 1, 2, 3])
def test_benchmark_rejects_unsupported_rows_before_setup(rows):
    with pytest.raises(ValueError, match='at least 4'):
        available_cases(2026, rows, 16)


@pytest.mark.parametrize('rows', [4, 5, 10])
@pytest.mark.parametrize('seed', [0, 42, 2026])
def test_small_supported_benchmark_checks_both_implementations(rows, seed):
    cases = available_cases(seed, rows, 3)
    assert len(cases) == 20
    for case in cases:
        case.check(case.operation())
        if case.baseline is not None:
            case.check(case.baseline())


def test_benchmark_cli_rejects_small_rows_without_writing(tmp_path):
    output = tmp_path / 'results.json'
    run = subprocess.run([sys.executable, '-B', 'benchmark/run.py', '--rows', '2',
                          '--output', str(output)], text=True, capture_output=True,
                         cwd=Path(__file__).resolve().parents[1])
    assert run.returncode == 2
    assert '--rows must be at least 4' in run.stderr
    assert 'Traceback' not in run.stderr
    assert not output.exists()


@pytest.mark.parametrize('function', [stats.var, stats.std])
@pytest.mark.parametrize('policy', ['propagate', 'raise', 'omit'])
def test_ddof_preserves_smallest_nonzero_result(function, policy):
    x = np.array([0., 0., 3e-162])
    smallest = np.nextafter(0., 1.)
    expected = smallest if function is stats.var else np.sqrt(smallest)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = function(x, ddof=2, nan_policy=policy)
    # Exact comparison: default allclose tolerance would accept zero here.
    assert result == expected and result > 0


@pytest.mark.parametrize('function', [stats.var, stats.std])
@pytest.mark.parametrize('last', [2e-162, 3e-162, 4e-162])
@pytest.mark.parametrize('ddof', [0, 1, 2])
def test_nearby_subnormal_results_match_direct_division(function, last, ddof):
    x = np.array([0., 0., last])
    reference = np.var(x, ddof=ddof)
    if function is stats.std:
        reference = np.sqrt(reference)
    assert function(x, ddof=ddof) == reference


@pytest.mark.parametrize('function', [stats.var, stats.std])
def test_subnormal_and_exhausted_slices_with_nan_omission(function):
    x = np.array([[0., 0.], [0., np.nan], [3e-162, 3e-162]])
    with pytest.warns(RuntimeWarning, match='ddof') as recorded:
        result = function(x, axis=0, ddof=2, keepdims=True, nan_policy='omit')
    assert len(recorded) == 1
    expected = np.nextafter(0., 1.)
    if function is stats.std:
        expected = np.sqrt(expected)
    assert result.shape == (1, 2)
    assert result[0, 0] == expected
    assert np.isnan(result[0, 1])


@pytest.mark.parametrize('function', [stats.var, stats.std])
@pytest.mark.parametrize('policy', ['propagate', 'raise', 'omit'])
@pytest.mark.parametrize('ddof', [3, 4])
def test_subnormal_exhausted_degrees_of_freedom(function, policy, ddof):
    with pytest.warns(RuntimeWarning, match='ddof') as recorded:
        result = function([0., 0., 3e-162], ddof=ddof, nan_policy=policy)
    assert len(recorded) == 1
    assert np.isnan(result)
