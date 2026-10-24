"""Member B: ordering, selection and tie rules."""

import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numcompute.sort_search import stable_sort, multi_key_sort, topk, quickselect, binary_search
from numcompute.rank import rank, percentile


class SortSearchTests(unittest.TestCase):
    def test_stable_sort_axes_and_noncontiguous_view(self):
        x = np.array([[3, 9, 1, 9], [2, 9, 0, 9]])[:, ::2]
        assert_array_equal(stable_sort(x), [[1, 3], [0, 2]])
        assert_array_equal(stable_sort(x, axis=0), [[2, 0], [3, 1]])
        assert_array_equal(stable_sort(x, axis=None), [0, 1, 2, 3])

    def test_stable_sort_nan_last(self):
        assert_allclose(stable_sort([np.nan, 2, 1]), [1, 2, np.nan], equal_nan=True)
        self.assertEqual(stable_sort([]).size, 0)

    def test_multi_key_priority_and_stable_ties(self):
        x = [[2, 0, 10], [1, 2, 11], [1, 1, 12], [1, 1, 13]]
        assert_array_equal(multi_key_sort(x, [0, 1], return_indices=True), [2, 3, 1, 0])
        with self.assertRaises(ValueError):
            multi_key_sort(x, [0, 0])

    def test_topk_ties_and_deterministic_indices(self):
        values, indices = topk([8, 8, 2, 9, 8], 3)
        assert_array_equal(values, [9, 8, 8])
        assert_array_equal(indices, [3, 0, 1])
        assert_array_equal(topk([3, 1, 1, 2], 2, largest=False)[1], [1, 2])

    def test_topk_extreme_k_and_all_equal(self):
        self.assertEqual(topk([], 0)[0].size, 0)
        assert_array_equal(topk([2, 2, 2], 2)[1], [0, 1])
        assert_array_equal(topk([1, 3, 2], 3, return_indices=False), [3, 2, 1])
        for k in (-1, 4):
            with self.assertRaises(ValueError):
                topk([1, 2, 3], k)
        with self.assertRaises(TypeError):
            topk([1, 2, 3], 1.5)

    def test_topk_preserves_large_unsigned_integers(self):
        x = np.array([2**64 - 2, 1, 2**64 - 1], dtype=np.uint64)
        assert_array_equal(topk(x, 2)[1], [2, 0])

    def test_quickselect_all_positions_and_no_mutation(self):
        x = np.array([9, 1, 3, 3, -1, 2])
        for k, expected in enumerate([-1, 1, 2, 3, 3, 9]):
            self.assertEqual(quickselect(x, k), expected)
        assert_array_equal(x, [9, 1, 3, 3, -1, 2])
        self.assertEqual(quickselect([5, 5, 5], 2), 5)

    def test_selection_nonfinite_and_empty_errors(self):
        for function in (quickselect, topk):
            with self.assertRaises(ValueError):
                function([1, np.nan], 1)
        with self.assertRaises(ValueError):
            quickselect([], 0)

    def test_binary_search_duplicates_and_boundaries(self):
        self.assertEqual(binary_search([1, 3, 3, 5], 3), (1, True))
        self.assertEqual(binary_search([1, 3, 5], 4), (2, False))
        self.assertEqual(binary_search([1, 3, 5], 0), (0, False))
        self.assertEqual(binary_search([1, 3, 5], 9), (3, False))
        self.assertEqual(binary_search([], 1), (0, False))


class RankTests(unittest.TestCase):
    def test_tie_methods(self):
        x = [30, 10, 10, 20]
        assert_array_equal(rank(x, "average"), [4, 1.5, 1.5, 3])
        assert_array_equal(rank(x, "dense"), [3, 1, 1, 2])
        assert_array_equal(rank(x, "ordinal"), [4, 1, 2, 3])

    def test_descending_rank_puts_largest_first(self):
        x = [70, 85, 85, 92]
        assert_array_equal(rank(x, ascending=False), [4, 2.5, 2.5, 1])
        assert_array_equal(rank(x, "dense", ascending=False), [3, 2, 2, 1])
        assert_array_equal(rank(x, "ordinal", ascending=False), [4, 2, 3, 1])
        big = np.array([2**64 - 1, 0, 2**64 - 2], dtype=np.uint64)
        assert_array_equal(rank(big, ascending=False), [1, 3, 2])
        assert_allclose(rank([3, np.nan, 1], ascending=False), [1, np.nan, 2], equal_nan=True)

    def test_all_equal_empty_and_nan_ranking(self):
        assert_array_equal(rank([4, 4, 4]), [2, 2, 2])
        self.assertEqual(rank([]).size, 0)
        assert_allclose(rank([3, np.nan, 1]), [2, np.nan, 1], equal_nan=True)
        self.assertTrue(np.isnan(rank([3, np.nan], nan_policy="propagate")).all())
        with self.assertRaises(ValueError):
            rank([np.nan], nan_policy="raise")

    def test_percentile_interpolations(self):
        for method, expected in [("linear", 7.5), ("lower", 0), ("higher", 10), ("midpoint", 5)]:
            self.assertEqual(percentile([0, 10, 20, 30], 25, method), expected)
        assert_array_equal(percentile([0, 10, np.nan, 20], [0, 50, 100]), [0, 10, 20])
        self.assertEqual(percentile([True, False, True, True], 50), 1.0)

    def test_percentile_axis_and_invalid_arguments(self):
        assert_array_equal(percentile([[0, 10], [10, 30]], 50, axis=0), [5, 20])
        for q in (-1, 101, np.nan):
            with self.assertRaises(ValueError):
                percentile([1, 2], q)
        with self.assertRaises(ValueError):
            rank([1, 2], "invented")


if __name__ == "__main__":
    unittest.main()
