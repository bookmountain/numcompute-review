"""Member C: axis-aware statistics, streaming moments, metrics and ROC."""

import contextlib
import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute import stats
from numcompute.stats import RunningStats
from numcompute.metrics import (confusion_matrix, accuracy, precision, recall, f1,
                                mse, roc_curve, auc, roc_auc_score)


@contextlib.contextmanager
def _no_warnings():
    """Fail the enclosed block if any warning is emitted."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


class DescriptiveStatsTests(unittest.TestCase):
    X = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 10.0]])

    def test_axis_none_int_tuple_and_keepdims_shapes(self):
        x = np.arange(24, dtype=float).reshape(2, 3, 4)
        for function in (stats.mean, stats.median, stats.var, stats.std, stats.min, stats.max):
            self.assertEqual(np.ndim(function(x)), 0)
            self.assertEqual(function(x, axis=0).shape, (3, 4))
            self.assertEqual(function(x, axis=-1).shape, (2, 3))
            self.assertEqual(function(x, axis=(0, 2)).shape, (3,))
            self.assertEqual(function(x, axis=1, keepdims=True).shape, (2, 1, 4))
            self.assertEqual(function(x, keepdims=True).shape, (1, 1, 1))

    def test_values_match_numpy_along_each_axis(self):
        for axis in (None, 0, 1):
            assert_allclose(stats.mean(self.X, axis=axis), np.mean(self.X, axis=axis))
            assert_allclose(stats.median(self.X, axis=axis), np.median(self.X, axis=axis))
            assert_allclose(stats.var(self.X, axis=axis, ddof=1), np.var(self.X, axis=axis, ddof=1))
            assert_allclose(stats.std(self.X, axis=axis), np.std(self.X, axis=axis))
            assert_array_equal(stats.min(self.X, axis=axis), np.min(self.X, axis=axis))
            assert_array_equal(stats.max(self.X, axis=axis), np.max(self.X, axis=axis))

    def test_noncontiguous_bool_and_integer_inputs(self):
        view = np.arange(20).reshape(4, 5)[::2, ::2]          # non-contiguous
        assert_array_equal(stats.mean(view, axis=1), [2.0, 12.0])
        self.assertEqual(stats.median([True, False, True]), 1.0)
        self.assertEqual(stats.min(np.array([5, 3], dtype=np.uint64)).dtype, np.uint64)
        big = np.array([2**64 - 1, 1], dtype=np.uint64)
        self.assertEqual(stats.max(big), 2**64 - 1)

    def test_all_equal_values_have_zero_spread(self):
        self.assertEqual(stats.var([4, 4, 4, 4]), 0.0)
        self.assertEqual(stats.std([4, 4, 4, 4]), 0.0)
        self.assertEqual(stats.median([4, 4, 4]), 4.0)

    def test_nan_policy_propagate_omit_and_raise(self):
        x = np.array([[1.0, np.nan], [3.0, 4.0]])
        assert_allclose(stats.mean(x, axis=0), [2.0, np.nan])                     # default
        assert_allclose(stats.mean(x, axis=0, nan_policy="omit"), [2.0, 4.0])
        assert_allclose(stats.median(x, axis=1, nan_policy="omit"), [1.0, 3.5])
        assert_allclose(stats.min(x, axis=0, nan_policy="omit"), [1.0, 4.0])
        assert_allclose(stats.var(x, axis=0, nan_policy="omit"), [1.0, 0.0])
        with self.assertRaises(ValueError):
            stats.mean(x, nan_policy="raise")
        with self.assertRaises(ValueError):
            stats.mean(x, nan_policy="drop")

    def test_all_nan_slice_under_omit_gives_nan_and_warns(self):
        x = np.array([[np.nan, 1.0], [np.nan, 3.0]])
        for function in (stats.mean, stats.median, stats.var, stats.std, stats.min, stats.max):
            with self.assertWarns(RuntimeWarning):
                result = function(x, axis=0, nan_policy="omit")
            self.assertTrue(np.isnan(result[0]))
            self.assertFalse(np.isnan(result[1]))
        with _no_warnings():  # a partially-NaN slice must not warn
            stats.mean(x, axis=1, nan_policy="omit")

    def test_ddof_and_insufficient_observations(self):
        assert_allclose(stats.var([2.0, 4.0], ddof=1), 2.0)
        with self.assertWarns(RuntimeWarning):
            self.assertTrue(np.isnan(stats.var([5.0], ddof=1)))
        with self.assertWarns(RuntimeWarning):
            out = stats.var([[np.nan, 1.0], [2.0, 3.0]], axis=0, ddof=1, nan_policy="omit")
        assert_allclose(out, [np.nan, 2.0])
        with self.assertRaises(TypeError):
            stats.var([1, 2], ddof=1.5)
        with self.assertRaises(ValueError):
            stats.var([1, 2], ddof=-1)

    def test_empty_infinite_and_non_numeric_inputs_are_rejected(self):
        for function in (stats.mean, stats.median, stats.var, stats.min, stats.max):
            with self.assertRaises(ValueError):
                function([])
            with self.assertRaises(ValueError):
                function([1.0, np.inf])
            with self.assertRaises(TypeError):
                function(["a", "b"])
        with self.assertRaises(ValueError):
            stats.mean(np.ones((2, 2)), axis=2)


class HistogramQuantileTests(unittest.TestCase):
    def test_histogram_counts_edges_and_density(self):
        counts, edges = stats.histogram([1, 2, 2, 3, 4], bins=3, range=(1, 4))
        assert_array_equal(counts, [1, 2, 2])
        assert_allclose(edges, [1, 2, 3, 4])
        counts, edges = stats.histogram([[1, 2], [2, 3]], bins=[0, 2, 4])   # 2-D is flattened
        assert_array_equal(counts, [1, 3])
        density, _ = stats.histogram([0.5, 1.5, 1.5, 1.5], bins=2, range=(0, 2), density=True)
        assert_allclose(density.sum() * 1.0, 1.0)

    def test_histogram_nan_policies(self):
        counts, _ = stats.histogram([1, np.nan, 2], bins=2, range=(0, 2))    # omit by default
        assert_array_equal(counts, [0, 2])
        counts, edges = stats.histogram([1, np.nan, 2], bins=2, range=(0, 2), nan_policy="propagate")
        self.assertTrue(np.isnan(counts).all())
        assert_allclose(edges, [0, 1, 2])
        with self.assertWarns(RuntimeWarning):
            counts, _ = stats.histogram([np.nan, np.nan], bins=2)
        assert_array_equal(counts, [0, 0])
        with self.assertRaises(ValueError):
            stats.histogram([1, np.nan], nan_policy="raise")

    def test_histogram_invalid_bins_and_range(self):
        with self.assertRaises(ValueError):
            stats.histogram([1, 2], bins=0)
        with self.assertRaises(TypeError):
            stats.histogram([1, 2], bins=2.5)
        with self.assertRaises(ValueError):
            stats.histogram([1, 2], bins=[0, 2, 1])
        with self.assertRaises(ValueError):
            stats.histogram([1, 2], range=(3, 1))
        with self.assertRaises(ValueError):
            stats.histogram([])

    def test_quantile_methods_vector_q_and_axes(self):
        x = [0, 10, 20, 30]
        for method, expected in [("linear", 7.5), ("lower", 0), ("higher", 10),
                                 ("midpoint", 5), ("nearest", 10)]:
            self.assertEqual(stats.quantile(x, 0.25, method=method), expected)
        assert_array_equal(stats.quantile(x, [0, 0.5, 1]), [0, 15, 30])
        m = np.arange(12, dtype=float).reshape(3, 4)
        self.assertEqual(stats.quantile(m, [0.1, 0.9], axis=0).shape, (2, 4))
        self.assertEqual(stats.quantile(m, 0.5, axis=1, keepdims=True).shape, (3, 1))
        assert_allclose(stats.quantile(m, 0.5, axis=1), stats.median(m, axis=1))

    def test_quantile_nan_handling_and_invalid_q(self):
        x = np.array([[1.0, np.nan], [3.0, np.nan], [2.0, 5.0]])
        assert_allclose(stats.quantile(x, 0.5, axis=0), [2.0, np.nan])
        assert_allclose(stats.quantile(x, 0.5, axis=0, nan_policy="omit"), [2.0, 5.0])
        with self.assertWarns(RuntimeWarning):
            out = stats.quantile([np.nan, np.nan], 0.5, nan_policy="omit")
        self.assertTrue(np.isnan(out))
        for q in (-0.1, 1.5, np.nan, [[0.5]]):
            with self.assertRaises(ValueError):
                stats.quantile([1, 2], q)
        with self.assertRaises(ValueError):
            stats.quantile([1, 2], 0.5, method="spline")


class RunningStatsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.data = rng.normal(loc=3.0, scale=2.0, size=(500, 3))
        self.data[rng.random(self.data.shape) < 0.1] = np.nan

    def test_uneven_chunks_equal_batch_statistics(self):
        rs = RunningStats()
        for start, stop in [(0, 1), (1, 7), (7, 7), (7, 250), (250, 500)]:
            rs.update(self.data[start:stop])
        assert_array_equal(rs.count, np.sum(~np.isnan(self.data), axis=0))
        assert_allclose(rs.mean, np.nanmean(self.data, axis=0))
        assert_allclose(rs.var(), np.nanvar(self.data, axis=0))
        assert_allclose(rs.var(ddof=1), np.nanvar(self.data, axis=0, ddof=1))
        assert_allclose(rs.std(), np.nanstd(self.data, axis=0))
        assert_allclose(rs.min, np.nanmin(self.data, axis=0))
        assert_allclose(rs.max, np.nanmax(self.data, axis=0))

    def test_merge_equals_single_pass_and_is_symmetric(self):
        a = RunningStats().update(self.data[:123])
        b = RunningStats().update(self.data[123:])
        whole = RunningStats().update(self.data)
        for merged in (a.merge(b), b.merge(a)):
            assert_array_equal(merged.count, whole.count)
            assert_allclose(merged.mean, whole.mean)
            assert_allclose(merged.var(ddof=1), whole.var(ddof=1), rtol=1e-12)
            assert_allclose(merged.min, whole.min)
            assert_allclose(merged.max, whole.max)
        # merge does not mutate either operand
        assert_array_equal(a.count, RunningStats().update(self.data[:123]).count)

    def test_merge_with_empty_accumulator_and_empty_batches(self):
        a = RunningStats().update([[1.0, 2.0], [3.0, 4.0]])
        empty = RunningStats()
        for merged in (a.merge(empty), empty.merge(a)):
            assert_allclose(merged.mean, [2.0, 3.0])
            assert_array_equal(merged.count, [2, 2])
        self.assertIsNone(RunningStats().update(np.empty((0, 4))).n_features)
        a.update(np.empty((0, 2)))
        assert_array_equal(a.count, [2, 2])
        with self.assertRaises(RuntimeError):
            _ = empty.mean

    def test_single_feature_and_single_observation(self):
        rs = RunningStats().update([5.0])
        assert_array_equal(rs.count, [1])
        assert_allclose(rs.mean, [5.0])
        assert_allclose(rs.var(), [0.0])
        self.assertTrue(np.isnan(rs.var(ddof=1)[0]))
        rs.update([7.0, 9.0])
        assert_allclose(rs.mean, [7.0])
        assert_allclose(rs.var(ddof=1), [4.0])
        assert_allclose([rs.min[0], rs.max[0]], [5.0, 9.0])

    def test_nan_policies_per_column(self):
        batch = np.array([[1.0, np.nan], [3.0, 4.0]])
        omit = RunningStats().update(batch)
        assert_array_equal(omit.count, [2, 1])
        assert_allclose(omit.mean, [2.0, 4.0])
        nothing = RunningStats().update([[np.nan], [np.nan]])
        assert_array_equal(nothing.count, [0])
        self.assertTrue(np.isnan(nothing.mean[0]))
        propagate = RunningStats(nan_policy="propagate").update(batch)
        assert_allclose(propagate.mean, [2.0, np.nan])
        self.assertTrue(np.isnan(propagate.var()[1]) and np.isnan(propagate.max[1]))
        with self.assertRaises(ValueError):
            RunningStats(nan_policy="raise").update(batch)

    def test_invalid_batches_and_merges(self):
        rs = RunningStats().update(np.ones((3, 2)))
        with self.assertRaises(ValueError):
            rs.update(np.ones((3, 5)))
        with self.assertRaises(ValueError):
            rs.update(np.ones((2, 2, 2)))
        with self.assertRaises(ValueError):
            rs.update([1.0, np.inf])
        with self.assertRaises(TypeError):
            rs.update(["x"])
        with self.assertRaises(TypeError):
            rs.merge(42)
        with self.assertRaises(ValueError):
            rs.merge(RunningStats().update(np.ones((1, 3))))
        with self.assertRaises(ValueError):
            rs.merge(RunningStats(nan_policy="propagate").update(np.ones((1, 2))))
        with self.assertRaises(ValueError):
            RunningStats(nan_policy="mean")


class ClassificationMetricsTests(unittest.TestCase):
    y_true = [0, 1, 2, 2, 0, 1]
    y_pred = [0, 2, 2, 2, 0, 0]

    def test_confusion_matrix_rows_are_true_columns_are_predicted(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        assert_array_equal(cm, [[2, 0, 0], [1, 0, 1], [0, 0, 2]])
        self.assertEqual(cm.dtype, np.int64)
        self.assertEqual(cm[1, 2], 1)   # true class 1 predicted as 2

    def test_confusion_matrix_explicit_label_order_and_absent_classes(self):
        cm = confusion_matrix(["cat", "dog"], ["dog", "dog"], labels=["dog", "cat", "bird"])
        assert_array_equal(cm, [[1, 0, 0], [1, 0, 0], [0, 0, 0]])
        assert_array_equal(confusion_matrix([True, False], [True, True]), [[0, 1], [0, 1]])
        with self.assertRaises(ValueError):
            confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])    # 2 is missing
        with self.assertRaises(ValueError):
            confusion_matrix([0, 1], [0, 1], labels=[0, 0, 1])

    def test_accuracy_and_binary_scores(self):
        self.assertAlmostEqual(accuracy(self.y_true, self.y_pred), 4 / 6)
        yt, yp = [0, 1, 1, 0, 1], [0, 1, 0, 0, 0]
        self.assertEqual(precision(yt, yp), 1.0)
        self.assertAlmostEqual(recall(yt, yp), 1 / 3)
        self.assertAlmostEqual(f1(yt, yp), 0.5)
        self.assertEqual(precision(yt, yp, pos_label=0), 0.5)
        self.assertEqual(recall(["a", "b", "b"], ["a", "b", "a"], pos_label="b"), 0.5)

    def test_multiclass_averaging(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # class 1 is never predicted
            assert_allclose(precision(self.y_true, self.y_pred, average=None), [2 / 3, 0.0, 2 / 3])
            assert_allclose(recall(self.y_true, self.y_pred, average=None), [1.0, 0.0, 1.0])
            assert_allclose(f1(self.y_true, self.y_pred, average=None), [0.8, 0.0, 0.8])
            self.assertAlmostEqual(precision(self.y_true, self.y_pred, average="macro"), 4 / 9)
            self.assertAlmostEqual(precision(self.y_true, self.y_pred, average="weighted"), 4 / 9)
            self.assertAlmostEqual(recall(self.y_true, self.y_pred, average="macro"), 2 / 3)
            self.assertAlmostEqual(f1(self.y_true, self.y_pred, average="macro"), 8 / 15)
        for function in (precision, recall, f1):
            self.assertAlmostEqual(function(self.y_true, self.y_pred, average="micro"), 4 / 6)
        with self.assertRaises(ValueError):
            precision(self.y_true, self.y_pred)                 # binary with 3 classes
        with self.assertRaises(ValueError):
            precision([0, 1], [0, 1], average="mean")

    def test_undefined_ratios_use_zero_division_and_warn(self):
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(precision([0, 1], [0, 0]), 0.0)   # no predicted positives
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(recall([0, 0], [0, 1]), 0.0)      # no true positives
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(precision([0, 1], [0, 0], zero_division=1.0), 1.0)
        with self.assertWarns(RuntimeWarning):
            self.assertTrue(np.isnan(precision([0, 1], [0, 0], zero_division=np.nan)))
        with self.assertWarns(RuntimeWarning):     # class 2 neither true nor predicted
            out = f1([0, 1], [0, 1], average=None, labels=[0, 1, 2])
        assert_array_equal(out, [1.0, 1.0, 0.0])
        with _no_warnings():                        # 0 recall with support is defined
            self.assertEqual(f1([0, 1], [0, 0], average="macro"), 1 / 3)
        with self.assertRaises(TypeError):
            precision([0, 1], [0, 1], zero_division="warn")

    def test_label_validation(self):
        with self.assertRaises(ValueError):
            confusion_matrix([0, 1], [0])
        with self.assertRaises(ValueError):
            confusion_matrix([], [])
        with self.assertRaises(ValueError):
            confusion_matrix([[0, 1]], [[0, 1]])
        with self.assertRaises(ValueError):
            confusion_matrix([0.0, np.nan], [0.0, 1.0])
        with self.assertRaises(TypeError):
            confusion_matrix(["a", "b"], [0, 1])
        with self.assertRaises(ValueError):
            precision([0, 1], [0, 1], pos_label=7)


class RegressionAndRocTests(unittest.TestCase):
    def test_mse_scalar_and_per_output(self):
        self.assertAlmostEqual(mse([1, 2, 3], [1, 2, 5]), 4 / 3)
        assert_allclose(mse([[1, 2], [3, 4]], [[1, 2], [5, 4]], axis=0), [2.0, 0.0])
        self.assertEqual(mse([2, 2], [2, 2]), 0.0)
        with self.assertRaises(ValueError):
            mse([1, 2], [1, 2, 3])
        with self.assertRaises(ValueError):
            mse([1, np.nan], [1, 2])
        with self.assertRaises(ValueError):
            mse([], [])

    def test_roc_curve_reference_points(self):
        fpr, tpr, thr = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        assert_allclose(fpr, [0, 0, 0.5, 0.5, 1])
        assert_allclose(tpr, [0, 0.5, 0.5, 1, 1])
        assert_allclose(thr, [np.inf, 0.8, 0.4, 0.35, 0.1])
        self.assertAlmostEqual(roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertEqual(roc_auc_score([0, 1], [0.2, 0.9]), 1.0)
        self.assertEqual(roc_auc_score([0, 1], [0.9, 0.2]), 0.0)
        self.assertEqual(roc_auc_score(["n", "p"], [0.2, 0.9], pos_label="p"), 1.0)

    def test_roc_score_ties_collapse_to_one_point(self):
        fpr, tpr, thr = roc_curve([0, 1, 0, 1], [0.5, 0.5, 0.2, 0.9])
        assert_allclose(thr, [np.inf, 0.9, 0.5, 0.2])      # 0.5 appears once
        assert_allclose(fpr, [0, 0, 0.5, 1])
        assert_allclose(tpr, [0, 0.5, 1, 1])
        self.assertAlmostEqual(roc_auc_score([0, 1, 0, 1], [0.5, 0.5, 0.2, 0.9]), 0.875)
        self.assertEqual(roc_auc_score([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)   # all tied
        fpr, tpr, thr = roc_curve([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(thr.size, 2)

    def test_roc_rejects_single_class_and_bad_inputs(self):
        with self.assertRaises(ValueError):
            roc_curve([1, 1, 1], [0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            roc_curve([0, 1, 2], [0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            roc_curve([0, 1], [0.1, 0.2], pos_label=5)
        with self.assertRaises(ValueError):
            roc_curve([0, 1], [0.1, np.nan])
        with self.assertRaises(ValueError):
            roc_curve([0, 1, 1], [0.1, 0.2])

    def test_auc_trapezoid_and_monotonic_check(self):
        self.assertEqual(auc([0, 0.5, 1], [0, 1, 1]), 0.75)
        self.assertEqual(auc([1, 0.5, 0], [1, 1, 0]), -0.75)   # signed, reversed x
        with self.assertRaises(ValueError):
            auc([0, 1, 0.5], [0, 1, 1])
        with self.assertRaises(ValueError):
            auc([0], [1])
        with self.assertRaises(ValueError):
            auc([0, 1], [1])


if __name__ == "__main__":
    unittest.main()
