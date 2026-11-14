"""Reusable matplotlib plots. Functions return (Figure, Axes).

Every plot accepts save_path=None and show=False; callers own and close figures.
Inputs are copied/coerced to 1D numeric arrays, except classification labels in
prediction plots, which may be strings. Each plot costs O(n) time/space in its
observations. Invalid shapes/lengths raise ValueError; NaN metric gaps are kept
so pre-training chunks are not presented as measured zero accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt


def _series(values):
    """Return metric values as a nonempty 1D float array without infinities."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size:
        raise ValueError("expected a nonempty 1D metric sequence")
    if np.isinf(values).any():
        raise ValueError("metrics must not contain infinite values")
    return values


def _finish(fig, ax, save_path, show):
    """Lay out the figure, then save and/or show it; return (fig, ax)."""
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_metric_over_time(
    metric_values, title="Streaming metric", ylabel="Accuracy", *, save_path=None, show=False
):
    """Plot (n_chunks,) values at 1-based chunk indexes; return (fig, ax)."""
    values = _series(metric_values)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(values) + 1), values, marker="o", markersize=3)
    ax.set(xlabel="Chunk", ylabel=ylabel, title=title)
    ax.grid(alpha=0.25)
    return _finish(fig, ax, save_path, show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), *, save_path=None, show=False):
    """Compare two (possibly unequal-length) streaming metric series."""
    a, b = _series(metric1), _series(metric2)
    if len(labels) != 2:
        raise ValueError("provide exactly two model labels")
    fig, ax = plt.subplots(figsize=(8, 4))
    for values, label in zip((a, b), labels):
        ax.plot(np.arange(1, len(values) + 1), values, label=label)
    ax.set(xlabel="Chunk", ylabel="Metric", title="Streaming model comparison")
    ax.legend()
    ax.grid(alpha=0.25)
    return _finish(fig, ax, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, *, save_path=None, show=False):
    """Plot aligned (n,) labels, with categorical ticks and marked mistakes."""
    t, p = np.asarray(y_true), np.asarray(y_pred)
    if t.ndim != 1 or t.shape != p.shape or not t.size:
        raise ValueError("true/predicted labels must be matching nonempty 1D arrays")
    if (t.dtype.kind in "US") != (p.dtype.kind in "US"):
        raise TypeError("true and predicted labels must share their kind")
    labels, codes = np.unique(np.concatenate((t, p)), return_inverse=True)
    n = len(t)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(np.arange(n), codes[:n], label="Ground truth", marker="o", alpha=0.6)
    ax.scatter(np.arange(n), codes[n:], label="Prediction", marker="x")
    errors = t != p
    ax.scatter(
        np.flatnonzero(errors),
        codes[n:][errors],
        facecolors="none",
        edgecolors="red",
        s=100,
        label="Error",
    )
    ax.set(
        xlabel="Sample in chunk",
        ylabel="Class",
        title="Predictions vs ground truth",
        yticks=np.arange(len(labels)),
        yticklabels=labels.astype(str),
    )
    ax.legend()
    return _finish(fig, ax, save_path, show)
