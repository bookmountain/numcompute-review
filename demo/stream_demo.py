"""Run the streaming CSV demo: python -m demo.stream_demo [--output DIR]."""

import argparse
import json
from pathlib import Path

import numpy as np

from numcompute_stream import (
    Pipeline,
    SimpleImputer,
    StandardScaler,
    DecisionTreeClassifier,
    StreamTrainer,
    StreamingStats,
)
from numcompute_stream.io import iter_csv_chunks

DATA = Path(__file__).with_name("data") / "stream.csv"


def make_trainer(buffer_size):
    """Construct an independent streaming pipeline and prequential evaluator."""
    return StreamTrainer(
        Pipeline(
            [
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
                (
                    "tree",
                    DecisionTreeClassifier(max_depth=5, buffer_size=buffer_size, random_state=2026),
                ),
            ]
        )
    )


def run_demo(path=DATA, chunk_size=100, buffer_sizes=(200, 500)):
    """Load custom CSV chunks once, update both trainers, return logs/statistics.

    Each chunk is tested against the previous model before training on it.
    Missing features are imputed; labels must be finite binary values.
    """
    trainers = [make_trainer(size) for size in buffer_sizes]
    stats = StreamingStats(window_size=500, bins=np.linspace(-4, 4, 17))
    latest = None
    for chunk in iter_csv_chunks(path, chunk_size=chunk_size, skip_header=1, missing="propagate"):
        X, raw_y = chunk[:, :-1], chunk[:, -1]
        if not np.isfinite(raw_y).all() or not np.isin(raw_y, [0, 1]).all():
            raise ValueError("demo CSV labels must be finite binary values")
        y = raw_y.astype(int)
        stats.update_stats(X)
        for trainer in trainers:
            trainer.fit_chunk(X, y)
        latest = X, y
    if latest is None:
        raise ValueError("CSV contained no data")
    return trainers, stats, latest


def summaries(trainers):
    """Return comparable scores on chunks where every model was already ready."""
    common = np.logical_and.reduce(
        [np.isfinite([r["chunk_accuracy"] for r in t.history]) for t in trainers]
    )
    rows = []
    for trainer in trainers:
        logs = trainer.history
        selected = [r for r, use in zip(logs, common) if use]
        n = sum(r["n_samples"] for r in selected)
        score = sum(r["n_samples"] * r["chunk_accuracy"] for r in selected) / n if n else None
        cumulative = logs[-1]["cumulative_accuracy"]
        rows.append(
            {
                "buffer_size": trainer.model.steps[-1][1].buffer_size,
                "retrains": trainer.model.steps[-1][1].n_retrains_,
                "pending_samples": trainer.model.n_buffered_,
                "evaluated_samples": trainer.metrics.result()["n_samples"],
                "prequential_accuracy": cumulative if np.isfinite(cumulative) else None,
                "common_evaluated_samples": n,
                "common_accuracy": score,
                "total_fit_seconds": sum(r["fit_seconds"] for r in logs),
                "final_memory_bytes": logs[-1]["memory_bytes"],
            }
        )
    return rows


def main():
    """Run the demo from the command line and optionally save the plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    trainers, stats, (X, y) = run_demo()
    print(json.dumps(summaries(trainers), indent=2))
    print("Latest 500-sample means:", stats.result()["mean"])
    if args.output:
        import matplotlib.pyplot as plt
        from numcompute_stream.visualise import (
            plot_metric_over_time,
            compare_models,
            plot_predictions_vs_ground_truth,
        )

        args.output.mkdir(parents=True, exist_ok=True)
        curves = [[r["chunk_accuracy"] for r in t.history] for t in trainers]
        plot_metric_over_time(
            curves[0],
            "Test-before-train accuracy",
            "Accuracy",
            save_path=args.output / "accuracy.png",
        )
        compare_models(
            *curves, labels=["Buffer 200", "Buffer 500"], save_path=args.output / "comparison.png"
        )
        plot_predictions_vs_ground_truth(
            y, trainers[0].model.predict(X), save_path=args.output / "predictions.png"
        )
        plt.close("all")


if __name__ == "__main__":
    main()
