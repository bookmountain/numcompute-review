"""Reuse the original CSV reader and chunk iterator."""

from numcompute.io import load_csv, iter_csv_chunks

__all__ = ["load_csv", "iter_csv_chunks"]
