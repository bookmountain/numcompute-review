"""Preprocessing/tree consistency, progressive evaluation and plot checks."""

import numpy as np
import pytest

from numcompute_stream import (
    Pipeline,
    StandardScaler,
    MinMaxScaler,
    SimpleImputer,
    OneHotEncoder,
    DecisionTreeClassifier,
    StreamTrainer,
    StreamingMetrics,
)


def make_pipe(buffer_size=4):
    return Pipeline(
        [
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("tree", DecisionTreeClassifier(buffer_size=buffer_size)),
        ]
    )


def test_pipeline_warmup_current_preprocessing_and_retrain():
    pipe = make_pipe()
    X, y = np.arange(4.0).reshape(-1, 1), np.array([0, 0, 1, 1])
    pipe.partial_fit(X[:2], y[:2])
    with pytest.raises(RuntimeError):
        pipe.predict(X)
    pipe.partial_fit(X[2:], y[2:])
    tree = pipe.named_steps["tree"]
    original_tree = tree.root_
    pipe.partial_fit([[1000.0], [np.nan]], [1, 1])
    current = pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(X))
    np.testing.assert_array_equal(pipe.predict(X), tree.predict(current))
    assert tree.root_ is original_tree and pipe.n_buffered_ == 2
    pipe.partial_fit([[1001.0], [1002.0]], [1, 1])
    assert tree.n_retrains_ == 2 and pipe.n_buffered_ == 0
    pipe.fit(X, y)
    np.testing.assert_array_equal(pipe.predict(X), y)
    assert tree.n_retrains_ == 0


def test_pipeline_encoding_requires_fixed_output_width():
    pipe = Pipeline(
        [
            ("encode", OneHotEncoder(handle_unknown="ignore")),
            ("tree", DecisionTreeClassifier(buffer_size=4)),
        ]
    )
    pipe.partial_fit(np.array([["b"], ["c"]]), [0, 1])
    pipe.partial_fit(np.array([["c"], ["b"]]), [1, 0])
    np.testing.assert_array_equal(pipe.predict(np.array([["b"], ["c"]])), [0, 1])
    with pytest.raises(ValueError, match="features"):
        pipe.partial_fit(np.array([["new"]]), [2])
    assert pipe.n_buffered_ == 0


def test_pipeline_buffer_owns_noncontiguous_input():
    backing = np.arange(16.0).reshape(4, 4)
    X = backing[:, ::2]
    pipe = Pipeline([("tree", DecisionTreeClassifier(buffer_size=4))])
    pipe.partial_fit(X[:2], [0, 0])
    backing[:2] = 999
    pipe.partial_fit(X[2:], [1, 1])
    assert pipe.predict([[0.0, 2.0]])[0] == 0


def test_transformer_only_pipeline():
    pipe = Pipeline([("scale", MinMaxScaler())])
    pipe.partial_fit([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(pipe.transform([[2.0, 3.0]]), [[0.5, 0.5]])
    pipe.partial_fit([[5.0, 6.0]])
    np.testing.assert_allclose(pipe.transform([[3.0, 4.0]]), [[0.5, 0.5]])


def test_pipeline_calls_public_incremental_estimator_with_transformed_chunks():
    class RecordingModel:
        def fit(self, X, y):
            return self.partial_fit(X, y)

        def partial_fit(self, X, y):
            self.X, self.y = X.copy(), y.copy()

        def predict(self, X):
            return np.full(len(X), self.y[0])

    model = RecordingModel()
    pipe = Pipeline([("scale", MinMaxScaler()), ("model", model)])
    pipe.partial_fit([[1.0], [3.0]], [1, 0])
    np.testing.assert_allclose(model.X, [[0.0], [1.0]])
    pipe.partial_fit([[5.0]], [0])
    np.testing.assert_allclose(model.X, [[1.0]])
    assert pipe.is_fitted_ and pipe.predict([[2.0]])[0] == 0


def test_stream_trainer_tests_before_training_and_skips_warmup():
    tree = DecisionTreeClassifier(buffer_size=2)
    trainer = StreamTrainer(tree)
    first = trainer.fit_chunk([[0], [1]], [0, 0])
    assert np.isnan(first["chunk_accuracy"]) and first["evaluated_samples"] == 0
    second = trainer.fit_chunk([[0], [1]], [1, 1])
    assert second["chunk_accuracy"] == 0 and second["cumulative_accuracy"] == 0
    assert second["ready"] and second["memory_bytes"] > 0 and second["fit_seconds"] >= 0
    third = trainer.score_chunk([[0], [1]], [1, 1])
    assert third["chunk_accuracy"] == 1 and third["cumulative_accuracy"] == 0.5
    assert tree.n_retrains_ == 2


def test_cumulative_accuracy_weighted_by_samples_not_chunks():
    tree = DecisionTreeClassifier(buffer_size=10).fit([[0]], [0])
    trainer = StreamTrainer(tree, metrics=StreamingMetrics(window_size=2))
    trainer.score_chunk([[0]], [1])
    record = trainer.score_chunk([[0]] * 3, [0, 0, 0])
    assert record["cumulative_accuracy"] == 0.75
    assert trainer.metrics.result()["accuracy"] == 1


def test_trainer_auc_uses_positive_class_column():
    pipe = make_pipe().fit(np.arange(4.0).reshape(-1, 1), [0, 0, 1, 1])
    trainer = StreamTrainer(pipe, metrics=StreamingMetrics(track_auc=True))
    trainer.score_chunk([[0.0], [3.0]], [0, 1])
    assert trainer.metrics.result()["roc_auc"] == 1


def test_csv_stream_integration(tmp_path):
    from numcompute_stream.io import iter_csv_chunks

    path = tmp_path / "data.csv"
    path.write_text("x,label\n0,0\n1,0\n2,1\n3,1\n4,1\n")
    trainer = StreamTrainer(make_pipe())
    for chunk in iter_csv_chunks(path, chunk_size=2, skip_header=1):
        trainer.fit_chunk(chunk[:, :-1], chunk[:, -1].astype(int))
    assert len(trainer.history) == 3 and trainer.history[-1]["evaluated_samples"] == 1


def test_predict_before_training_does_not_mutate_metrics():
    trainer = StreamTrainer(make_pipe())
    with pytest.raises(RuntimeError):
        trainer.score_chunk([[0.0]], [0])
    assert not trainer.history and trainer.metrics.result()["n_samples"] == 0


@pytest.mark.parametrize("which", ["metric", "compare", "predictions"])
def test_visualisations_save_headlessly(tmp_path, which):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from numcompute_stream.visualise import (
        plot_metric_over_time,
        compare_models,
        plot_predictions_vs_ground_truth,
    )

    path = tmp_path / f"{which}.png"
    if which == "metric":
        fig, ax = plot_metric_over_time([np.nan, 0.5, 0.75], "Accuracy", "Accuracy", save_path=path)
        assert np.isnan(ax.lines[0].get_ydata()[0])
    elif which == "compare":
        fig, ax = compare_models([0.5, 0.6], [0.4, 0.9], ["Small", "Large"], save_path=path)
        assert len(ax.lines) == 2
    else:
        fig, ax = plot_predictions_vs_ground_truth(["cat", "dog"], ["dog", "dog"], save_path=path)
    assert path.stat().st_size > 1000
    plt.close(fig)


def test_demo_with_one_oversized_chunk_has_json_safe_unmeasured_scores():
    import json
    from demo.stream_demo import run_demo, summaries

    trainers, _, _ = run_demo(chunk_size=2000)
    result = summaries(trainers)
    assert all(row["prequential_accuracy"] is None for row in result)
    assert all(row["common_evaluated_samples"] == 0 for row in result)
    json.dumps(result, allow_nan=False)


def test_generic_incremental_estimator_evaluates_after_first_update():
    class ConstantModel:
        def partial_fit(self, X, y):
            self.value = y[0]

        def predict(self, X):
            return np.full(len(X), self.value)

    trainer = StreamTrainer(ConstantModel())
    trainer.fit_chunk([[0.0]], [1])
    record = trainer.fit_chunk([[1.0]], [1])
    assert record["chunk_accuracy"] == 1 and record["evaluated_samples"] == 1
