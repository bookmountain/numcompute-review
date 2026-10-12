# Assignment requirements map

This is a concise, paraphrased working checklist derived from a locally saved
printout of the university course page and the accompanying
`Assignment2.1-specs.md`. The PDF filename is
`Assignment 2.1_ Programming Task 1.pdf`; it is a personal printout saved for
reference. The live course page and instructions from course staff take
precedence. The printout displays 26 October 2026 and states 9:00 am on the
Monday after the mid-semester break; check the live page for changes.

## Scope and constraints

- Team of 3–4; assignment weighting 30%.
- Build a modular scientific-computing toolkit for the Weeks 1–6 scope.
- Use plain Python and NumPy only; do not use pandas, scikit-learn, PyTorch, or
  another external data/ML library.
- Prefer vectorised NumPy calculations. Retain loops only where the algorithm,
  arbitrary callable, streaming I/O, or orchestration genuinely requires them.
- Define consistent axes, shapes, NaN/tie semantics, useful type/shape errors,
  and numerically stable calculations.
- Public docstrings must cover parameters, returns, shapes, raised exceptions,
  and time/space complexity.

## Required package functionality

| Module | Required work | Notes from the course-page printout and core specification |
|---|---|---|
| `io.py` | Numeric CSV loading, custom delimiters, missing-value skip/fill, NumPy output | PDF also calls for chunking/streaming and dtype handling |
| `preprocessing.py` | StandardScaler, MinMaxScaler, OneHotEncoder, fitted transform API | Imputation is optional in the core Markdown but used by the required demo |
| `sort_search.py` | Stable and multi-key sorting, argpartition top-k, quickselect, binary search | Top-k returns indexes by default; binary search returns insertion index and existence |
| `rank.py` | Average/dense/ordinal ranks and linear/lower/higher/midpoint percentiles | Tie behaviour must be explicit |
| `stats.py` | Mean, median, standard deviation, min/max, histogram, quantiles, axes | PDF additionally names variance and streaming Welford statistics |
| `metrics.py` | Accuracy, precision, recall, F1, confusion matrix, MSE | ROC/AUC is marked bonus |
| `optim.py` | Finite-difference gradient and Jacobian | Line search is optional; demo asks for scalar and vector differentiation |
| `pipeline.py` | Transformer/Estimator protocols and Pipeline chaining | Core spec says no ML model implementation is required; PDF also names composition and feature unions |
| `utils.py` | Distances, activations, logsumexp, top-k helpers, batching | Emphasise stability and documented memory limits |
| `benchmarking.py` | Reusable microbenchmark harness and loop-versus-NumPy comparisons | Record reproducibility and environment information |

## Required notebook demonstration

`demo/quickstart.ipynb` must demonstrate all of the following:

1. Read a CSV through `io.py`, including rows with missing values.
2. Preprocess with a scaler, imputer, and one-hot encoder.
3. Show top-k, quickselect, binary search, tie-aware ranking, and percentiles.
4. Compute mean/variance, a histogram, and quantiles.
5. Estimate scalar gradients and vector Jacobians with finite differences.
6. Compare vectorised implementations with Python loops and include a short
   performance table.

The PDF also requires the video to show setup/running the demo, use of the
pipeline, and interpretation of outputs such as metrics, ranks, and gradients.

## Testing and benchmarking acceptance

- The integrated repository must contain at least 20 unit tests.
- Required edge cases include empty arrays, all-equal values, duplicates/ties,
  extreme `k`, NaNs, and non-contiguous strides.
- Benchmark inputs must be reproducible, vectorised and loop outputs must be
  checked for agreement, and timing/environment details must be recorded.
- Timing ratios are evidence from a specific environment, not universal claims
  and not pass/fail unit-test assertions.

## Submission package

1. Git repository link containing all modules, tests, notebook, benchmark
   material, README usage instructions, and packaging configuration.
2. Demonstration video of at most five minutes, submitted as MP4 or a shareable
   link.
3. Technical report PDF of at most four pages covering design/vectorisation,
   APIs, testing/edge cases, benchmark results, and teamwork reflection.

## Rubric evidence

| Criterion | Points | Evidence the group should retain |
|---|---:|---|
| Algorithm design | 20 | Algorithm explanations, complexity, and justified choices |
| Functionality implementation | 20 | Correct behaviour against the module contracts |
| Code reusability | 15 | Clear reusable functions/classes and abstractions |
| Code maintainability | 15 | Systematic debugging, tests, and understandable structure |
| Testing and debugging evidence | 10 | Meaningful normal/error/edge-case coverage |
| Teamwork | 10 | Consistent integration and identifiable individual work |
| Style, documentation, comments | 10 | Consistent style and complete useful documentation |
