"""Run currently implemented workloads: python benchmark/run.py --help."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

# Support direct execution from any working directory without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from numcompute.benchmarking import BenchmarkCase, run_benchmarks
from numcompute import metrics, optim, pipeline, stats, utils


def _checker(expected, *, rtol=1e-10, atol=1e-12):
    def check(actual):
        if np.shape(actual) != np.shape(expected):
            raise AssertionError(f"shape mismatch: {np.shape(actual)} != {np.shape(expected)}")
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return check


def _loop_sigmoid(x):
    return np.array([1 / (1 + math.exp(-float(v))) if v >= 0 else
                     math.exp(float(v)) / (1 + math.exp(float(v)))
                     for v in x.flat]).reshape(x.shape)


def _loop_softmax(x):
    rows = []
    for row in x:
        maximum = max(row)
        weights = [math.exp(float(v - maximum)) for v in row]
        total = sum(weights)
        rows.append([v / total for v in weights])
    return np.array(rows)


def _loop_logsumexp(x):
    result = []
    for row in x:
        maximum = max(row)
        result.append(maximum + math.log(sum(math.exp(float(v - maximum)) for v in row)))
    return np.array(result)


def _loop_var(x, ddof=1):
    """Two-pass per-column variance with explicit Python loops."""
    rows, columns = x.shape
    result = []
    for j in range(columns):
        total = 0.0
        for i in range(rows):
            total += float(x[i, j])
        mean = total / rows
        squares = 0.0
        for i in range(rows):
            squares += (float(x[i, j]) - mean) ** 2
        result.append(squares / (rows - ddof))
    return np.array(result)


def _loop_welford(x):
    """Per-element Welford update of count, mean and M2 for each column."""
    rows, columns = x.shape
    count = [0] * columns
    mean = [0.0] * columns
    m2 = [0.0] * columns
    for i in range(rows):
        for j in range(columns):
            value = float(x[i, j])
            count[j] += 1
            delta = value - mean[j]
            mean[j] += delta / count[j]
            m2[j] += delta * (value - mean[j])
    return np.array([m2[j] / (count[j] - 1) for j in range(columns)])


def _loop_histogram(values, edges):
    """Count values into right-closed-last bins by scanning the edges."""
    counts = [0] * (len(edges) - 1)
    last = len(edges) - 2
    for value in values:
        value = float(value)
        if value < edges[0] or value > edges[-1]:
            continue
        index = last
        for k in range(last):
            if value < edges[k + 1]:
                index = k
                break
        counts[index] += 1
    return np.array(counts)


def _loop_confusion(y_true, y_pred, n_classes):
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        matrix[t][p] += 1
    return np.array(matrix)


class _Activation:
    """Benchmark-only adapter around an existing utility, with no learned state."""

    def __init__(self, function):
        self.function = function

    def fit(self, X):
        return self

    def transform(self, X):
        return self.function(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def available_cases(seed, rows, features):
    """Prepare current cases with a local RNG; no missing module APIs are assumed.

    At least four rows and positive features define float64 input shape.
    Four rows are needed to guarantee coverage of all classification labels.
    Returns a list of BenchmarkCase; NumPy propagates invalid seed/shape errors.
    Setup takes O(rows*features + d**2) time/space, where d=min(features, 16),
    excluding retained reference outputs of the same order.
    """
    if rows < 4:
        raise ValueError("rows must be at least 4 for four-class metrics benchmarks")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, features))
    y = rng.normal(size=x.shape)
    metadata = {"shape": list(x.shape), "dtype": str(x.dtype),
                "distribution": "standard_normal", "layout": "C-contiguous",
                "rtol": 1e-10, "atol": 1e-12}
    cases = []

    def compare(name, operation, baseline, label="python_loop", notes=""):
        cases.append(BenchmarkCase(name, operation, _checker(baseline()), baseline,
                                   label, {**metadata, "notes": notes}))

    compare("utils.sigmoid", lambda: utils.sigmoid(x), lambda: _loop_sigmoid(x))
    compare("utils.relu", lambda: utils.relu(x),
            lambda: np.array([max(float(v), 0.0) for v in x.flat]).reshape(x.shape))
    compare("utils.softmax", lambda: utils.softmax(x, axis=1), lambda: _loop_softmax(x))
    compare("utils.logsumexp", lambda: utils.logsumexp(x, axis=1),
            lambda: _loop_logsumexp(x))
    compare("utils.euclidean_distance", lambda: utils.euclidean_distance(x, y),
            lambda: np.array([math.hypot(*(float(a) - float(b) for a, b in zip(rx, ry)))
                              for rx, ry in zip(x, y)]))
    compare("utils.manhattan_distance", lambda: utils.manhattan_distance(x, y),
            lambda: np.array([sum(abs(float(a) - float(b)) for a, b in zip(rx, ry))
                              for rx, ry in zip(x, y)]))

    batch_size = 128
    # Setup-only validation: copying/concatenating contents is not timed.
    original_rows = x.copy()
    batches = list(utils.iter_batches(x, batch_size))
    np.testing.assert_array_equal(np.concatenate(batches, axis=0), original_rows)
    np.testing.assert_array_equal(x, original_rows)
    del batches, original_rows

    def consume_batches():
        return np.array([batch.shape[0] for batch in utils.iter_batches(x, batch_size)])
    compare("utils.iter_batches", consume_batches,
            lambda: np.array([x[start:start + batch_size].shape[0]
                              for start in range(0, rows, batch_size)]),
            "python_slice_loop", "Fully consume batch views; return batch lengths; batch_size=128. "
            "Exact batch contents/order and input preservation checked during untimed setup.")

    point = rng.normal(size=min(features, 16))
    for name, operation, expected in (
        ("optim.grad", lambda: optim.grad(lambda z: np.sum(z*z), point), 2 * point),
        ("optim.jacobian", lambda: optim.jacobian(lambda z: z*z, point), np.diag(2 * point)),
    ):
        cases.append(BenchmarkCase(name, operation, _checker(expected, rtol=1e-6, atol=1e-7),
                     metadata={"shape": list(point.shape), "dtype": str(point.dtype),
                               "method": "central", "step": 1e-5,
                               "rtol": 1e-6, "atol": 1e-7,
                               "notes": "Analytic derivative check only; no timed baseline."}))

    def sequential():
        return pipeline.Pipeline([("relu", _Activation(utils.relu)),
                                  ("sigmoid", _Activation(utils.sigmoid))])
    def parallel():
        return pipeline.FeatureUnion([("relu", _Activation(utils.relu)),
                                      ("sigmoid", _Activation(utils.sigmoid))])
    fitted = sequential().fit(x)
    union = parallel().fit(x)
    for name, operation, direct in (
        ("pipeline.Pipeline.transform", lambda: fitted.transform(x),
         lambda: utils.sigmoid(utils.relu(x))),
        ("pipeline.Pipeline.fit_transform", lambda: sequential().fit_transform(x),
         lambda: utils.sigmoid(utils.relu(x))),
        ("pipeline.FeatureUnion.transform", lambda: union.transform(x),
         lambda: np.concatenate((utils.relu(x), utils.sigmoid(x)), axis=1)),
        ("pipeline.FeatureUnion.fit_transform", lambda: parallel().fit_transform(x),
         lambda: np.concatenate((utils.relu(x), utils.sigmoid(x)), axis=1)),
    ):
        compare(name, operation, direct, "direct_numpy_utilities",
                "Benchmark-only activation adapters; fit is a no-op. fit_transform includes "
                "fresh construction; transform uses pre-fitted objects. Ratio measures "
                "composition/copying overhead, not a vectorisation speedup.")
    # Statistics: column-wise reductions, streaming moments and binning.
    compare("stats.mean", lambda: stats.mean(x, axis=0),
            lambda: np.array([sum(float(v) for v in x[:, j]) / rows for j in range(features)]),
            notes="Per-column mean of a finite float64 matrix; nan_policy='propagate'.")
    compare("stats.var", lambda: stats.var(x, axis=0, ddof=1), lambda: _loop_var(x),
            notes="Per-column sample variance (ddof=1), two-pass loop baseline.")
    compare("stats.RunningStats.update", lambda: stats.RunningStats().update(x).var(ddof=1),
            lambda: _loop_welford(x), "python_welford_loop",
            "One batched Chan/Welford update of a fresh accumulator versus a per-element "
            "Welford loop; both return the per-column sample variance.")
    edges = np.linspace(-4.0, 4.0, 33)
    compare("stats.histogram", lambda: stats.histogram(x, bins=edges)[0],
            lambda: _loop_histogram(x.ravel(), edges.tolist()), "python_bin_scan_loop",
            "32 fixed bins on [-4, 4] over the flattened matrix; returns counts only. "
            "Baseline scans the edge list per value.")

    # Metrics: integer labels from a seeded 4-class problem and binary scores.
    n_classes = 4
    y_true = rng.integers(0, n_classes, size=rows)
    # Guarantee all classes, and hence both ROC classes, even at minimum size.
    y_true[:n_classes] = np.arange(n_classes)
    rng.shuffle(y_true)
    y_pred = np.where(rng.random(rows) < 0.7, y_true, rng.integers(0, n_classes, size=rows))
    label_metadata = {"shape": [rows], "dtype": str(y_true.dtype), "n_classes": n_classes,
                      "agreement": 0.7, "layout": "C-contiguous", "rtol": 1e-10, "atol": 1e-12}
    cases.append(BenchmarkCase(
        "metrics.confusion_matrix", lambda: metrics.confusion_matrix(y_true, y_pred),
        _checker(_loop_confusion(y_true, y_pred, n_classes)),
        lambda: _loop_confusion(y_true, y_pred, n_classes), "python_loop",
        {**label_metadata, "notes": "4x4 matrix, rows=true, columns=predicted; labels inferred."}))

    def loop_macro_f1():
        cm = _loop_confusion(y_true, y_pred, n_classes)
        scores = []
        for k in range(n_classes):
            tp = cm[k][k]
            fp = sum(cm[i][k] for i in range(n_classes)) - tp
            fn = sum(cm[k][j] for j in range(n_classes)) - tp
            scores.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        return sum(scores) / n_classes
    cases.append(BenchmarkCase(
        "metrics.f1_macro", lambda: metrics.f1(y_true, y_pred, average="macro"),
        _checker(loop_macro_f1()), loop_macro_f1, "python_loop",
        {**label_metadata, "notes": "Macro-averaged F1 over four classes; every class present."}))

    y_binary = (y_true == 0).astype(int)
    y_score = np.round(rng.random(rows) * 0.5 + 0.5 * y_binary, 2)   # rounding creates ties

    def loop_roc_auc():
        positives = [float(s) for s, t in zip(y_score, y_binary) if t == 1]
        negatives = [float(s) for s, t in zip(y_score, y_binary) if t == 0]
        wins = 0.0
        for p in positives:
            for n in negatives:
                wins += 1.0 if p > n else (0.5 if p == n else 0.0)
        return wins / (len(positives) * len(negatives))
    cases.append(BenchmarkCase(
        "metrics.roc_auc_score", lambda: metrics.roc_auc_score(y_binary, y_score),
        _checker(loop_roc_auc()), loop_roc_auc, "python_pairwise_loop",
        {"shape": [rows], "dtype": str(y_score.dtype), "positives": int(y_binary.sum()),
         "score_decimals": 2, "rtol": 1e-10, "atol": 1e-12,
         "notes": "Sorted-threshold ROC AUC with tied scores collapsed, checked against the "
                  "O(P*N) Mann-Whitney pairwise count with ties scored one half."}))
    return cases


def main():
    """Parse CLI arguments, run prepared cases and save JSON; errors propagate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--number", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark/results.json")
    args = parser.parse_args()
    for name in ("rows", "features", "repeats", "number"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.rows < 4:
        parser.error("--rows must be at least 4 for four-class metrics benchmarks")
    if args.seed < 0 or args.warmups < 0:
        parser.error("--seed and --warmups must be nonnegative")
    report = run_benchmarks(available_cases(args.seed, args.rows, args.features),
                            seed=args.seed, repeats=args.repeats, number=args.number,
                            warmups=args.warmups)
    report["config"].update(rows=args.rows, features=args.features,
                            rng="numpy.random.default_rng / PCG64")
    sources = [Path(__file__).resolve(), ROOT / "numcompute/benchmarking.py",
               ROOT / "numcompute/_validation.py"]
    sources.extend(Path(module.__file__) for module in (utils, optim, pipeline, stats, metrics))
    report["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sources}
    try:
        report["git"] = {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        }
    except (OSError, subprocess.CalledProcessError):
        report["git"] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{'Case':43} {'Operation ms':>13} {'Baseline ms':>13} {'Base/op':>9}")
    for row in report["results"]:
        baseline = row.get("baseline")
        baseline_ms = f"{1000 * baseline['median_seconds']:.4f}" if baseline else "—"
        ratio = row.get("baseline_over_operation")
        ratio_text = f"{ratio:.2f}x" if ratio is not None else "—"
        print(f"{row['name']:43} {1000 * row['operation']['median_seconds']:13.4f} "
              f"{baseline_ms:>13} {ratio_text:>9}")
    print(f"All correctness checks passed. Results: {args.output}")
    print("Ratios describe this workload and environment only; see JSON baseline labels.")


if __name__ == "__main__":
    main()
