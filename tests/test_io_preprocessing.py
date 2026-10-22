"""Tests for CSV loading and preprocessing."""

import unittest
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np

from numcompute.io import load_csv, iter_csv_chunks
from numcompute.preprocessing import StandardScaler, MinMaxScaler



class TestLoadCSV(unittest.TestCase):
    """Check CSV values, shapes, and missing-value handling."""

    def test_load_numeric_csv(self):
        """Load numeric values without changing their arrangement."""
        csv = StringIO("1,2\n3,4\n")

        result = load_csv(csv)

        np.testing.assert_array_equal(
            result,
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        )

    def test_fill_missing_values(self):
        """Replace an empty field with the chosen fill value."""
        csv = StringIO("1,\n3,4\n")

        result = load_csv(csv, missing="fill", fill_value=0)

        np.testing.assert_array_equal(
            result,
            np.array([[1.0, 0.0], [3.0, 4.0]]),
        )

    def test_skip_rows_with_missing_values(self):
        """Remove a row containing an empty field."""
        csv = StringIO("1,\n3,4\n")

        result = load_csv(csv, missing="skip")

        np.testing.assert_array_equal(
            result,
            np.array([[3.0, 4.0]]),
        )

    def test_propagate_missing_values(self):
        """Keep an empty field as NaN."""
        csv = StringIO("1,\n3,4\n")

        result = load_csv(csv, missing="propagate")

        np.testing.assert_allclose(
            result,
            np.array([[1.0, np.nan], [3.0, 4.0]]),
            equal_nan=True,
        )


class TestCSVChunks(unittest.TestCase):
    """Check chunk sizes and consistent CSV handling."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "sample.csv"

    def test_chunk_sizes_and_values(self):
        """Read every record, including the smaller final chunk."""
        self.path.write_text(
            "1,2\n3,4\n5,6\n7,8\n9,10\n",
            encoding="utf-8",
        )

        chunks = list(iter_csv_chunks(self.path, chunk_size=2))

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].shape, (2, 2))
        self.assertEqual(chunks[1].shape, (2, 2))
        self.assertEqual(chunks[2].shape, (1, 2))
        np.testing.assert_array_equal(
            np.concatenate(chunks, axis=0),
            np.arange(1, 11).reshape(5, 2),
        )

    def test_header_delimiter_and_missing_values(self):
        """Skip the header once and fill missing fields."""
        self.path.write_text(
            "first;second\n1;\n\n3;4\n5;6\n",
            encoding="utf-8",
        )

        chunks = list(
            iter_csv_chunks(
                self.path,
                chunk_size=1,
                delimiter=";",
                skip_header=1,
                fill_value=0,
            )
        )

        np.testing.assert_array_equal(
            np.concatenate(chunks, axis=0),
            np.array([[1, 0], [3, 4], [5, 6]]),
        )

    def test_skip_empty_output_chunks(self):
        """Continue reading when a chunk contains only missing rows."""
        self.path.write_text(
            "1,\n,2\n3,4\n",
            encoding="utf-8",
        )

        chunks = list(
            iter_csv_chunks(
                self.path,
                chunk_size=2,
                missing="skip",
            )
        )

        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(chunks[0], [[3, 4]])

    def test_reject_changed_column_count(self):
        """Reject inconsistent fields across separate chunks."""
        self.path.write_text(
            "1,2\n3,4,5\n",
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            list(iter_csv_chunks(self.path, chunk_size=1))

    def test_reject_invalid_chunk_size(self):
        """Reject zero and Boolean chunk sizes."""
        with self.assertRaises(ValueError):
            list(iter_csv_chunks(self.path, chunk_size=0))

        with self.assertRaises(TypeError):
            list(iter_csv_chunks(self.path, chunk_size=True))


class TestStandardScaler(unittest.TestCase):
    """Check scaling and reuse of training statistics."""

    def test_scale_training_data(self):
        """Centre each feature and give it unit standard deviation."""
        X = np.array([[1, 10], [3, 20], [5, 30]], dtype=float)

        scaler = StandardScaler()
        result = scaler.fit_transform(X)

        np.testing.assert_allclose(
            result.mean(axis=0), [0, 0], atol=1e-12
        )
        np.testing.assert_allclose(result.std(axis=0), [1, 1])
        self.assertEqual(result.shape, X.shape)

    def test_reuse_training_statistics(self):
        """Transform new data without learning from it."""
        scaler = StandardScaler()
        self.assertIs(scaler.fit([[1, 10], [3, 20]]), scaler)

        result = scaler.transform([[5, 30]])

        np.testing.assert_allclose(result, [[3, 3]])
        np.testing.assert_allclose(scaler.mean_, [2, 15])
        np.testing.assert_allclose(scaler.scale_, [1, 5])

    def test_constant_feature(self):
        """Use a scale of one for a constant training feature."""
        scaler = StandardScaler()
        result = scaler.fit_transform([[2, 1], [2, 3]])

        np.testing.assert_allclose(result, [[0, -1], [0, 1]])
        np.testing.assert_allclose(
            scaler.transform([[4, 5]]), [[2, 3]]
        )

    def test_missing_values(self):
        """Ignore NaN during fitting and preserve it in the output."""
        X = np.array([[1, np.nan], [3, 10], [5, 20]])

        scaler = StandardScaler()
        result = scaler.fit_transform(X)

        np.testing.assert_allclose(scaler.mean_, [3, 15])
        self.assertTrue(np.isnan(result[0, 1]))
        np.testing.assert_allclose(result[1:, 1], [-1, 1])

    def test_reject_all_nan_feature(self):
        """Reject a training feature with no observed values."""
        with self.assertRaises(ValueError):
            StandardScaler().fit([[1, np.nan], [2, np.nan]])

    def test_noncontiguous_input_is_unchanged(self):
        """Accept an array view without changing its source data."""
        source = np.arange(24, dtype=float).reshape(4, 6)
        original = source.copy()
        X = source[:, ::2]

        result = StandardScaler().fit_transform(X)

        self.assertEqual(result.shape, X.shape)
        np.testing.assert_array_equal(source, original)
        np.testing.assert_allclose(
            result.mean(axis=0), [0, 0, 0], atol=1e-12
        )
        np.testing.assert_allclose(result.std(axis=0), [1, 1, 1])

    def test_require_fit_and_matching_features(self):
        """Reject use before fitting and incorrect feature counts."""
        scaler = StandardScaler()

        with self.assertRaises(RuntimeError):
            scaler.transform([[1, 2]])

        scaler.fit([[1, 2], [3, 4]])

        with self.assertRaises(ValueError):
            scaler.transform([[1, 2, 3]])

        self.assertEqual(
            scaler.transform(np.empty((0, 2))).shape, (0, 2)
        )


class TestMinMaxScaler(unittest.TestCase):
    """Check feature ranges and reuse of training limits."""

    def test_scale_training_data(self):
        """Map training minima to zero and maxima to one."""
        X = np.array([[1, 10], [3, 20], [5, 30]], dtype=float)

        result = MinMaxScaler().fit_transform(X)

        np.testing.assert_allclose(
            result,
            [[0, 0], [0.5, 0.5], [1, 1]],
        )
        self.assertEqual(result.shape, X.shape)

    def test_reuse_training_limits(self):
        """Use training limits even for values outside their range."""
        scaler = MinMaxScaler()
        self.assertIs(scaler.fit([[1, 10], [5, 30]]), scaler)

        result = scaler.transform([[7, 0]])

        np.testing.assert_allclose(result, [[1.5, -0.5]])
        np.testing.assert_allclose(scaler.data_min_, [1, 10])
        np.testing.assert_allclose(scaler.data_max_, [5, 30])
        np.testing.assert_allclose(scaler.data_range_, [4, 20])

    def test_constant_feature(self):
        """Avoid division by zero for a constant training feature."""
        scaler = MinMaxScaler()
        result = scaler.fit_transform([[2, 1], [2, 3]])

        np.testing.assert_allclose(result, [[0, 0], [0, 1]])
        np.testing.assert_allclose(
            scaler.transform([[4, 5]]), [[2, 2]]
        )

    def test_missing_values(self):
        """Ignore NaN when learning limits and preserve it afterward."""
        X = np.array([[1, np.nan], [3, 10], [5, 20]])

        scaler = MinMaxScaler()
        result = scaler.fit_transform(X)

        np.testing.assert_allclose(scaler.data_min_, [1, 10])
        np.testing.assert_allclose(scaler.data_max_, [5, 20])
        np.testing.assert_allclose(
            result,
            [[0, np.nan], [0.5, 0], [1, 1]],
            equal_nan=True,
        )

    def test_reject_all_nan_feature(self):
        """Reject a training feature with no observed values."""
        with self.assertRaises(ValueError):
            MinMaxScaler().fit([[1, np.nan], [2, np.nan]])

    def test_noncontiguous_input_is_unchanged(self):
        """Scale an array view without changing its source data."""
        source = np.arange(24, dtype=float).reshape(4, 6)
        original = source.copy()
        X = source[:, ::2]

        result = MinMaxScaler().fit_transform(X)

        np.testing.assert_array_equal(source, original)
        self.assertEqual(result.shape, X.shape)
        np.testing.assert_allclose(result.min(axis=0), [0, 0, 0])
        np.testing.assert_allclose(result.max(axis=0), [1, 1, 1])

    def test_require_fit_and_matching_features(self):
        """Reject use before fitting and incorrect feature counts."""
        scaler = MinMaxScaler()

        with self.assertRaises(RuntimeError):
            scaler.transform([[1, 2]])

        scaler.fit([[1, 2], [3, 4]])

        with self.assertRaises(ValueError):
            scaler.transform([[1, 2, 3]])

        self.assertEqual(
            scaler.transform(np.empty((0, 2))).shape, (0, 2)
        )


if __name__ == "__main__":
    unittest.main()
