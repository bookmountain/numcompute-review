"""This module uses NumPy to load numeric CSV files and handle missing values."""

import numpy as np


def load_csv(
    path,
    delimiter=",",
    skip_header=0,
    dtype=float,
    missing="fill",
    fill_value=np.nan,
):
    """Load a numeric CSV file into a 2D NumPy array.

    Parameters
    ----------
    path : str or path-like
        Path to the CSV file.
    delimiter : str, default=","
        Character used to separate values in each row.
    skip_header : int, default=0
        Number of lines to skip at the beginning of the file.
    dtype : NumPy-compatible numeric dtype, default=float
        Numeric type to use for the output array.
    missing : {"fill", "skip", "propagate"}, default="fill"
        How to handle missing values: replace them with fill_value,
        skip rows containing them, or keep them as NaN.
    fill_value : number, default=np.nan
        Value used to replace missing entries when missing="fill".

    Returns
    -------
    numpy.ndarray
        A 2D array with one row per remaining CSV record and
        one column per field.

    Notes
    -----
    Integer arrays cannot store NaN. For missing values, either
    choose an integer fill_value, use missing="skip", or select
    a floating-point dtype such as float.
    """
    if missing not in ("fill", "skip", "propagate"):
        raise ValueError(
            "missing must be 'fill', 'skip', or 'propagate'"
        )

    if isinstance(skip_header, (bool, np.bool_)) or not isinstance(
        skip_header, (int, np.integer)
    ):
        raise TypeError("skip_header must be an integer")

    if skip_header < 0:
        raise ValueError("skip_header must be zero or greater")

    dtype = np.dtype(dtype)

    if dtype.kind not in "iuf":
        raise TypeError("dtype must be a real numeric type")

    data = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=skip_header,
        dtype=dtype,
        comments=None,
        autostrip=True,
        usemask=True,
        loose=False,
        invalid_raise=True,
        ndmin=2,
        encoding="utf-8",
    )

    if data.size == 0:
        raise ValueError("CSV file contains no data")

    values = np.array(data.data, copy=True)
    missing_mask = np.ma.getmaskarray(data) | np.isnan(values)

    if missing == "skip":
        return values[~missing_mask.any(axis=1)]

    if missing_mask.any():
        replacement = np.nan if missing == "propagate" else fill_value

        if dtype.kind in "iu":
            if np.ndim(replacement) != 0:
                raise ValueError("fill_value must be a single number")

            limits = np.iinfo(dtype)
            if (
                not np.isfinite(replacement)
                or replacement != np.floor(replacement)
                or replacement < limits.min
                or replacement > limits.max
            ):
                raise ValueError(
                    "Missing-value replacement cannot be represented "
                    "by the integer dtype; use dtype=float or choose "
                    "a valid integer fill_value"
                )

        values[missing_mask] = replacement

    return values
