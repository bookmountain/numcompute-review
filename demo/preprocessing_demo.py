"""Demonstrate preprocessing with synthetic product data.

Columns: price (dollars), weight (grams), category_code (category label).
The first six records are training data; the last two are test data.
This small example demonstrates behaviour, not model performance.
"""

from pathlib import Path

import numpy as np

from numcompute.io import load_csv
from numcompute.preprocessing import (
    SimpleImputer,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)


def main():
    """Load the sample, preprocess it and display output shapes."""
    path = Path(__file__).parent / "data" / "sample.csv"
    data = load_csv(path, skip_header=1, missing="propagate")

    # Split before learning any replacement values or scaling statistics.
    train = data[:6]
    test = data[6:]

    numeric_train = train[:, :2]
    numeric_test = test[:, :2]
    category_train = train[:, 2:3]
    category_test = test[:, 2:3]

    # Learn replacements from training data and reuse them for test data.
    imputer = SimpleImputer(strategy="mean")
    filled_train = imputer.fit_transform(numeric_train)
    filled_test = imputer.transform(numeric_test)

    standard = StandardScaler()
    scaled_train = standard.fit_transform(filled_train)
    scaled_test = standard.transform(filled_test)

    # Demonstrate min-max scaling as an alternative to standard scaling.
    minmax = MinMaxScaler()
    minmax_train = minmax.fit_transform(filled_train)
    minmax_test = minmax.transform(filled_test)

    # Category 2 is unseen during training and becomes an all-zero block.
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoded_train = encoder.fit_transform(category_train)
    encoded_test = encoder.transform(category_test)

    # Combine standardised numeric features with encoded categories.
    prepared_train = np.concatenate(
        (scaled_train, encoded_train), axis=1
    )
    prepared_test = np.concatenate(
        (scaled_test, encoded_test), axis=1
    )

    print("Input columns: price, weight, category_code")
    print("Loaded data:", data.shape)
    print("Numeric train/test:", numeric_train.shape, numeric_test.shape)
    print("Imputed train/test:", filled_train.shape, filled_test.shape)
    print("Standard-scaled train/test:",
          scaled_train.shape, scaled_test.shape)
    print("Min-max-scaled train/test:",
          minmax_train.shape, minmax_test.shape)
    print("Encoded train/test:", encoded_train.shape, encoded_test.shape)
    print("Combined train/test:", prepared_train.shape, prepared_test.shape)
    print("Combined columns: scaled_price, scaled_weight, category_0, category_1")
    print("Training replacement values:", imputer.statistics_)
    print("Learned categories:", encoder.categories_[0])
    print("Encoded test categories:")
    print(encoded_test)
    print("Prepared test data:")
    print(np.round(prepared_test, 3))

    assert data.shape == (8, 3)
    assert prepared_train.shape == (6, 4)
    assert prepared_test.shape == (2, 4)
    assert np.isfinite(prepared_train).all()
    assert np.isfinite(prepared_test).all()
    np.testing.assert_allclose(imputer.statistics_, [24, 200])
    np.testing.assert_array_equal(encoded_test, [[0, 0], [1, 0]])

    print("Demo checks passed.")


if __name__ == "__main__":
    main()
