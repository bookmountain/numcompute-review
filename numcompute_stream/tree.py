"""Depth-limited CART classification tree with buffered retraining for streams.

Every split is a numeric question of the form ``X[:, feature] <= threshold``.
Features must be finite real numbers: impute missing values and one-hot encode
categories first, for example inside a ``Pipeline``.
"""

from dataclasses import dataclass

import numpy as np

from numcompute._validation import integer
from .metrics import _labels_array, accuracy

# A split must lower the impurity by more than this to count as an improvement.
_MIN_IMPROVEMENT = 1e-12


@dataclass
class _Node:
    """One node of the tree. A leaf keeps ``feature == -1``."""

    counts: np.ndarray  # training samples of each class that reached this node
    feature: int = -1
    threshold: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None

    @property
    def is_leaf(self):
        """True when this node has no split."""
        return self.feature < 0


class DecisionTreeClassifier:
    """Classification tree built from scratch with NumPy.

    Parameters
    ----------
    max_depth : int, default 5
        Maximum depth. The root has depth 0, so 0 gives a single leaf.
    min_samples_split : int, default 2
        A node with fewer samples than this becomes a leaf.
    max_features : None, int, "sqrt" or "log2", default None
        Number of features sampled at random at each node. None uses all.
    buffer_size : int, default 500
        Rows that ``partial_fit`` collects before the tree is retrained.
    criterion : "gini" or "entropy", default "gini"
        Impurity measure used to choose splits.
    random_state : int or None, default 0
        Seed for feature sampling, so results are reproducible.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted class labels known to the current tree.
    root_ : _Node or None
        Root of the current tree, or None before the first training.
    n_buffered_ : int
        Rows waiting in the buffer for the next retrain.
    n_retrains_ : int
        Buffered retrains since the last call to ``fit``.

    Notes
    -----
    X has shape (n_samples, n_features) and y has shape (n_samples,).
    Bad shapes or values raise ValueError, unsupported types raise TypeError,
    and predicting before any training raises RuntimeError.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        max_features=None,
        buffer_size=500,
        criterion="gini",
        random_state=0,
    ):
        self.max_depth = integer(max_depth, "max_depth")
        self.min_samples_split = integer(min_samples_split, "min_samples_split", 2)
        self.buffer_size = integer(buffer_size, "buffer_size", 1)
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if isinstance(max_features, str):
            if max_features not in ("sqrt", "log2"):
                raise ValueError("max_features must be None, an integer, 'sqrt' or 'log2'")
        elif max_features is not None:
            max_features = integer(max_features, "max_features", 1)
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.root_ = None
        self._buffer = []
        self.n_buffered_ = 0
        self.n_retrains_ = 0

    @property
    def is_fitted_(self):
        """True once a tree has been trained, even while a new buffer fills."""
        return self.root_ is not None

    # ------------------------------------------------------------------
    # Input checks
    # ------------------------------------------------------------------
    def _check_features(self, X, *, training=False):
        """Return X as a finite, real-valued (n_samples, n_features) array."""
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] == 0 or (training and X.shape[0] == 0):
            raise ValueError(
                "X must have shape (n_samples, n_features) with nonempty training data"
            )
        if X.dtype.kind not in "biuf":
            raise TypeError("X must contain real numbers; one-hot encode categories first")
        if not np.isfinite(X).all():
            raise ValueError("X must be finite; impute missing values before training")
        return X

    def _check_training_data(self, X, y):
        """Validate a training pair and return it as NumPy arrays."""
        X = self._check_features(X, training=True)
        y = _labels_array(y, "y")
        if y.size != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if isinstance(self.max_features, int) and self.max_features > X.shape[1]:
            raise ValueError("max_features exceeds the input feature count")
        return X, y

    def _check_consistent(self, X, y=None):
        """Reject data whose width or label kind differs from earlier data."""
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        if y is not None and (y.dtype.kind in "US") != self._string_labels:
            raise TypeError("label kind must match previous chunks")

    # ------------------------------------------------------------------
    # Growing a tree
    # ------------------------------------------------------------------
    def _impurity(self, counts):
        """Gini or entropy of class counts, for one row or many rows at once."""
        p = counts / np.sum(counts, axis=-1, keepdims=True)
        if self.criterion == "gini":
            return 1 - np.sum(p * p, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.sum(p * np.log2(np.where(p > 0, p, 1)), axis=-1)

    def _should_stop(self, n_samples, counts, depth):
        """Stop rules: depth limit reached, too few samples, or a pure node."""
        return (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.count_nonzero(counts) == 1
        )

    def _candidate_features(self, n_features):
        """Feature indexes to search at one node: all, or a random subset."""
        if self.max_features is None:
            return np.arange(n_features)
        if self.max_features == "sqrt":
            size = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            size = max(1, int(np.log2(n_features)))
        else:
            size = self.max_features
        return np.sort(self._rng.choice(n_features, size, replace=False))

    def _best_split(self, X, codes, features):
        """Return the (feature, threshold) with the lowest weighted impurity.

        For each feature the rows are sorted once. A cumulative sum of one-hot
        class rows then gives the left-side class counts of every threshold
        at the same time, so there is no Python loop over thresholds.
        Returns None when no split lowers the impurity.
        """
        n_samples = len(X)
        counts = np.bincount(codes, minlength=self.classes_.size)
        one_hot = np.eye(counts.size, dtype=np.int64)
        best_score, best_split = float(self._impurity(counts)), None
        for feature in features:
            order = np.argsort(X[:, feature], kind="stable")
            values = X[order, feature]
            # A threshold is only useful where the sorted value changes.
            boundaries = np.flatnonzero(values[:-1] != values[1:])
            if boundaries.size == 0:
                continue
            left_counts = np.cumsum(one_hot[codes[order]], axis=0)[boundaries]
            n_left = boundaries + 1
            scores = (
                n_left * self._impurity(left_counts)
                + (n_samples - n_left) * self._impurity(counts - left_counts)
            ) / n_samples
            winner = int(np.argmin(scores))
            if scores[winner] < best_score - _MIN_IMPROVEMENT:
                best_score = float(scores[winner])
                # The observed lower value is used, not a midpoint, so very large
                # integers cannot overflow or round.
                best_split = (int(feature), values[boundaries[winner]])
        return best_split

    def _build(self, X, codes, depth):
        """Grow the subtree for these rows recursively and return its root."""
        counts = np.bincount(codes, minlength=self.classes_.size)
        node = _Node(counts)
        if self._should_stop(len(X), counts, depth):
            return node
        split = self._best_split(X, codes, self._candidate_features(X.shape[1]))
        if split is None:
            return node
        node.feature, node.threshold = split
        goes_left = X[:, node.feature] <= node.threshold
        node.left = self._build(X[goes_left], codes[goes_left], depth + 1)
        node.right = self._build(X[~goes_left], codes[~goes_left], depth + 1)
        return node

    def _train(self, X, y):
        """Replace the current tree with a new one trained on X and y."""
        self.n_features_in_ = X.shape[1]
        self._string_labels = y.dtype.kind in "US"
        self.classes_, codes = np.unique(y, return_inverse=True)
        self._rng = np.random.default_rng(self.random_state)
        self.root_ = self._build(X, codes, depth=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Train immediately on X and y, discard any buffered rows, return self."""
        X, y = self._check_training_data(X, y)
        self._train(X, y)
        self._buffer = []
        self.n_buffered_ = 0
        self.n_retrains_ = 0
        return self

    def partial_fit(self, X, y):
        """Add one chunk to the buffer and retrain once it holds buffer_size rows.

        The chunk is copied, so later changes to the caller's arrays have no
        effect. A retrain uses every buffered row, then empties the buffer.
        Before the first retrain there is no tree and ``predict`` raises
        RuntimeError. Between retrains ``predict`` uses the latest tree.
        Returns self.
        """
        X, y = self._check_training_data(X, y)
        if hasattr(self, "n_features_in_"):
            self._check_consistent(X, y)
        else:
            self.n_features_in_ = X.shape[1]
            self._string_labels = y.dtype.kind in "US"
        self._buffer.append((X.copy(), y.copy()))
        self.n_buffered_ += y.size
        if self.n_buffered_ >= self.buffer_size:
            self._retrain_from_buffer()
        return self

    def _retrain_from_buffer(self):
        """Train a new tree on all buffered rows, then empty the buffer."""
        X = np.concatenate([chunk_X for chunk_X, _ in self._buffer])
        y = np.concatenate([chunk_y for _, chunk_y in self._buffer])
        self._train(X, y)
        self.n_retrains_ += 1
        self._buffer.clear()
        self.n_buffered_ = 0

    def predict_proba(self, X):
        """Return class probabilities of shape (n_samples, n_classes).

        Columns follow ``classes_``. Each row holds the class shares of the
        leaf that the sample reaches.
        """
        if not self.is_fitted_:
            raise RuntimeError("tree is not trained; fill the buffer or call fit()")
        X = self._check_features(X)
        self._check_consistent(X)
        probabilities = np.empty((X.shape[0], self.classes_.size))
        self._route(self.root_, X, np.arange(X.shape[0]), probabilities)
        return probabilities

    def _route(self, node, X, rows, out):
        """Send these rows down from node and write leaf class shares into out.

        All rows travel together as an index array, so each node is visited
        once instead of once per sample.
        """
        if rows.size == 0:
            return
        if node.is_leaf:
            out[rows] = node.counts / node.counts.sum()
            return
        goes_left = X[rows, node.feature] <= node.threshold
        self._route(node.left, X, rows[goes_left], out)
        self._route(node.right, X, rows[~goes_left], out)

    def predict(self, X):
        """Return the most likely class of each row, shape (n_samples,).

        Ties go to the first class in ``classes_``, which is the smallest label.
        """
        probabilities = self.predict_proba(X)  # raises RuntimeError before training
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X, y):
        """Return the accuracy of the current tree on X and y."""
        return accuracy(y, self.predict(X))
