"""Recreate stream.csv using NumPy only (seed 2026, no external dataset)."""

from pathlib import Path
import numpy as np


def make_data(rows=1600, seed=2026):
    """Return synthetic numeric features and binary labels with a midpoint drift."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 3))
    threshold = np.where(np.arange(rows) < rows // 2, 0.0, 0.8)
    y = (X[:, 0] + 0.6 * X[:, 1] > threshold).astype(int)
    y ^= (rng.random(rows) < 0.03).astype(int)
    X[rng.random(X.shape) < 0.015] = np.nan
    return X, y


def main():
    """Write demo/data/stream.csv and print its path."""
    X, y = make_data()
    path = Path(__file__).with_name("data") / "stream.csv"
    path.parent.mkdir(exist_ok=True)
    np.savetxt(
        path,
        np.column_stack((X, y)),
        delimiter=",",
        header="x0,x1,x2,label",
        comments="",
        fmt=["%.6f", "%.6f", "%.6f", "%d"],
    )
    print(path)


if __name__ == "__main__":
    main()
