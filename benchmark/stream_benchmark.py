"""Compare loop/NumPy split search and four buffers on the same CSV stream.

Fit timings include preprocessing, buffering and retraining.
Run: python -m benchmark.stream_benchmark
"""

import json
from pathlib import Path
from statistics import median

import numpy as np

from demo.stream_demo import DATA, run_demo, summaries
from numcompute_stream.io import iter_csv_chunks
from numcompute_stream import SimpleImputer, StandardScaler, DecisionTreeClassifier
from numcompute.benchmarking import BenchmarkCase, run_benchmarks as benchmark

BUFFER_SIZES = (100, 200, 500, 1000)


def loop_best_split(tree, X, codes, features):
    """Try each threshold separately, recounting the classes on both sides."""
    counts = np.bincount(codes, minlength=tree.classes_.size)
    best, split = float(tree._impurity(counts)), None
    for feature in features:
        values = X[:, feature]
        for threshold in np.unique(values)[:-1]:
            mask = values <= threshold
            nl = np.count_nonzero(mask)
            if nl == 0 or nl == len(X):
                continue
            left = np.bincount(codes[mask], minlength=counts.size)
            right = np.bincount(codes[~mask], minlength=counts.size)
            score = (nl * tree._impurity(left) + (len(X) - nl) * tree._impurity(right)) / len(X)
            if score < best - 1e-12:
                best, split = float(score), (int(feature), threshold)
    return split


def split_case(X, y):
    # Fit a leaf to initialise class codes without timing tree construction.
    """Build a benchmark case comparing loop and vectorised split search on X, y."""
    tree = DecisionTreeClassifier(max_depth=0).fit(X, y)
    _, codes = np.unique(y, return_inverse=True)
    features = np.arange(X.shape[1])
    expected = loop_best_split(tree, X, codes, features)

    def check(split):
        """Fail if the vectorised split differs from the loop result."""
        if split != expected:
            raise AssertionError(f"split mismatch: {split!r} != {expected!r}")

    return BenchmarkCase(
        f"split_search_{len(X)}",
        operation=lambda: tree._best_split(X, codes, features),
        baseline=lambda: loop_best_split(tree, X, codes, features),
        check=check,
        metadata={"rows": len(X), "features": X.shape[1], "criterion": "gini"},
    )


def run_benchmarks(repeats=5, inner=3, chunk_size=100):
    """Run the split-search and buffer-size benchmarks and return the results."""
    if repeats < 1 or inner < 1 or chunk_size < 1:
        raise ValueError("repeats, inner, and chunk_size must be positive")
    data = np.concatenate(list(iter_csv_chunks(DATA, chunk_size=chunk_size, skip_header=1)))
    X = StandardScaler().fit_transform(SimpleImputer().fit_transform(data[:, :-1]))
    y = data[:, -1].astype(int)
    cases = [split_case(X[:size], y[:size]) for size in BUFFER_SIZES]
    measured = benchmark(cases, repeats=repeats, number=inner)
    by_repeat = [
        summaries(run_demo(chunk_size=chunk_size, buffer_sizes=BUFFER_SIZES)[0])
        for _ in range(repeats)
    ]
    comparison = []
    for i, result in enumerate(by_repeat[-1]):
        row = result.copy()
        row["median_fit_seconds"] = median(run[i]["total_fit_seconds"] for run in by_repeat)
        comparison.append(row)
    return {
        "seed": 2026,
        "rows": len(y),
        "chunk_size": chunk_size,
        "repeats": repeats,
        "inner": inner,
        "environment": measured["environment"],
        "evaluation": "test-before-train; common_accuracy uses identical scored chunks",
        "memory": "approximate model-owned Python/NumPy bytes, not process RSS",
        "split_search": measured["results"],
        "buffers": comparison,
    }


def main():
    """Run the benchmarks, save stream_results.json and print it."""
    result = run_benchmarks()
    text = json.dumps(result, indent=2, allow_nan=False)
    Path(__file__).with_name("stream_results.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
