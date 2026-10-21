# NumCompute responsibilities and acceptance criteria

Replace pending names with actual names and GitHub usernames. This file records
assigned responsibility; completed work and evidence belong in
`CONTRIBUTIONS.md`. The working requirements summary is
`ASSIGNMENT_REQUIREMENTS.md`.

## Member A — data loading and preprocessing

**Branch:** `data-preprocessing`

**Owned files:**

- `numcompute/io.py`
- `numcompute/preprocessing.py`
- `tests/test_io_preprocessing.py`
- `demo/data/sample.csv`

**Must achieve:**

- Numeric CSV loading with delimiter, header, dtype, fill, skip, and propagate
  options, plus bounded-size chunk iteration.
- `StandardScaler`, `MinMaxScaler`, `SimpleImputer`, and `OneHotEncoder` using
  fitted `fit`/`transform` behaviour without train/test leakage.
- Tests for missing fields, malformed rows, constant and all-NaN columns,
  non-contiguous inputs, unknown categories, and reuse of training statistics.
- A quickstart section that loads the shared CSV, preprocesses numeric and
  categorical columns, and explains the resulting shapes.
- The data/API section of the report and the first section of the video.

**Additional coordination:** maintain installation/package instructions and
provide the shared sample dataset required by the notebook.

## Shushan — algorithms

**Branch:** `book/algorithm`

**Owned files:**

- `numcompute/sort_search.py`
- `numcompute/rank.py`
- `tests/test_sort_rank.py`

**Must achieve:**

- Stable sorting, multi-key row sorting, deterministic top-k selection,
  `np.argpartition`-based top-k, educational quickselect, and leftmost binary
  search returning both insertion position and existence.
- Average, dense, and ordinal tie-aware ranking plus percentile interpolation.
- Tests for ties, duplicate values, empty/extreme `k`, unsigned integers,
  non-contiguous arrays, NaN policies, interpolation methods, and input
  non-mutation.
- A quickstart section using a numeric column supplied by A's CSV loader and
  showing expected indexes, ranks, and percentile results.
- The algorithm/complexity section of the report and the second video section.

**Additional coordination:** combine and format the final report after every
member supplies their reviewed section.

## Member C — statistics, metrics, and benchmarking

**Branch:** `stats-metrics`

**Owned files:**

- `numcompute/stats.py`
- `numcompute/metrics.py`
- `numcompute/benchmarking.py`
- `tests/test_stats_metrics.py`
- `benchmark/run.py` and recorded benchmark results

**Must achieve:**

- Axis-aware descriptive statistics, histograms, quantiles, and mergeable
  Welford/Chan streaming statistics with explicit NaN policies. Include mean,
  median, variance, standard deviation, minimum, and maximum.
- Accuracy, precision, recall, F1, confusion matrix, MSE, ROC, and AUC with
  documented label order and multiclass averaging.
- Correctness-checked, seeded, repeated NumPy-versus-loop benchmarks with
  environment metadata and no timing assertions in unit tests.
- Tests for axes and shapes, all-NaN slices, streaming merges, undefined ratios,
  multiclass averaging, score ties, and invalid inputs.
- Statistics/metrics/benchmark quickstart examples, the benchmark/report
  evidence, and the third video section.

## Ruofan Li — composition, numerical methods, and integration

**Branch:** `pipeline-optim`

**Owned files:**

- `numcompute/pipeline.py`
- `numcompute/optim.py`
- `numcompute/utils.py`
- final integration of `numcompute/__init__.py`
- `tests/test_optim_pipeline_utils.py`
- `demo/quickstart.py`
- `demo/quickstart.ipynb`
- `demo/check_notebook.py`

**Must achieve:**

- Transformer/estimator protocols, sequential pipelines, feature unions, and
  composition/feature unions, plus column-based transformation where useful,
  with shape and fitted-state checks. No ML model implementation is required by
  the core specification.
- Finite-difference gradients/Jacobians, optional line search, stable
  activations/logsumexp, distances, and batching.
- Tests using analytic derivatives, estimator test doubles, large logits,
  invalid shapes, feature mismatches, and integration paths.
- Assemble—not silently author—the four member-supplied quickstart sections;
  run the script and every notebook cell from a fresh environment.
- The composition/numerical-stability report section and fourth video section.

**Additional coordination:** create the integration branch after reviewed module
pull requests are merged and run the final package, test, notebook, and link
checks.

## Cross-review assignments

| Reviewer | Reviews | Evidence required |
|---|---|---|
| A | Shushan | Pull-request review covering algorithm correctness and tests |
| Shushan | C | Pull-request review covering shapes, metrics, and benchmarks |
| C | Ruofan | Pull-request review covering protocols and numerical stability |
| Ruofan | A | Pull-request review covering CSV/preprocessing contracts |

## Integration sequence

1. Every member accepts the GitHub invitation and creates their branch from
   `main` using their own Git identity.
2. Members implement, test, document, and push only their allocated files.
3. The assigned reviewer checks each pull request before it enters `main`.
4. A adds the agreed sample CSV; all members supply runnable quickstart snippets
   and expected output to D.
5. D creates an integration branch from the reviewed `main`, builds the script
   and notebook, and resolves public exports in `numcompute/__init__.py`.
6. C runs the final reproducible benchmarks and updates the performance table.
7. B combines the four reviewed report sections; all members verify accuracy.
8. The group records the video, verifies repository/video access, completes
   `CONTRIBUTIONS.md`, nominates one submitter, and runs all checks before
   submission.

## Final group checklist

- [ ] Pending names and GitHub usernames have been replaced.
- [ ] Each member's scoped tests and the complete suite pass.
- [ ] The integrated suite contains at least 20 meaningful unit tests.
- [ ] Every pull request has its assigned cross-review.
- [ ] The shared CSV exercises missing values, ties, and categories.
- [ ] The quickstart script and every notebook cell run in order.
- [ ] Benchmarks record seed, repetitions, timings, and environment.
- [ ] README/API documentation matches the integrated code.
- [ ] Actual contributions are recorded accurately.
- [ ] The report is at most four pages and the video is at most five minutes.
- [ ] A nominated member will submit before the live Canvas deadline.
- [ ] Repository and video links are accessible to the marker.
