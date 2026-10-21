"""Tests for CSV loading and preprocessing."""

import unittest
from io import StringIO

import numpy as np

from numcompute.io import load_csv


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


if __name__ == "__main__":
    unittest.main()
