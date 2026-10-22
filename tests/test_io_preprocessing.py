"""Tests for CSV loading and preprocessing."""

import unittest
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np

from numcompute.io import load_csv, iter_csv_chunks


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



if __name__ == "__main__":
    unittest.main()
