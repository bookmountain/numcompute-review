## Pipelines, optimisation, utilities and benchmarking (Raven)

### Design and implementation

These modules use Python and NumPy. `grad` and `jacobian` support forward, backward and central finite differences. For n inputs, one-sided schemes require n + 1 function evaluations; central schemes require 2n for gradients and 2n + 1 for Jacobians. For smooth functions, one-sided schemes have O(h) truncation error and central schemes have O(h²) truncation error. Validation rejects invalid arguments, inconsistent output shapes, ineffective perturbations and non-finite inputs, evaluations or derivatives, including overflow. Float64 copies preserve caller inputs.

`Transformer` and `Estimator` protocols define composition interfaces. `Pipeline` fits steps sequentially, forwards targets only to a final estimator, and reuses fitted state for transformation or prediction. `FeatureUnion` fits branches on independent input copies and concatenates features in branch order. Containers check fitted state, unique nonempty names, callable methods, two-dimensional arrays and row preservation. Step objects are fitted directly. There is no global fixed feature-count restriction: `n_features_in_` is metadata; compatibility belongs to individual steps.

`sigmoid` uses sign-dependent evaluation; `relu` applies an elementwise maximum. `softmax` and `logsumexp` subtract maxima before exponentiation; the latter handles empty and non-finite slices. Python-integer temporaries preserve integer differences, with a rounding-residual correction in `logsumexp`. `euclidean_distance` uses `np.hypot.reduce` to avoid unnecessary overflow; `manhattan_distance` sums absolute differences. Two integer distance inputs are subtracted before float conversion, avoiding wraparound and lost gaps. `iter_batches` yields ordered views whose modification can affect the source. The duplicate `top_k_indices` was removed; integrated `sort_search.topk` is the project's single top-k implementation.

Benchmarks exclude preparation and correctness checks, including batch contents/order and input preservation, from timing. Warmups precede repeated calls measured with `perf_counter_ns`; elapsed nanoseconds become seconds per call. Comparison order alternates, and raw samples plus minimum, median and maximum timings are retained. The integrated `benchmark/results.json` snapshot records 20 workloads with seed 2026, one warmup, five repeats and three calls per sample. It records shapes, tolerances, configuration and environment metadata. The default and four-row benchmark runs each passed all 20 correctness checks.

Below, m is derivative output size, N input elements, Q reduction output elements, Nᵢ pipeline intermediate sizes, b branches, M concatenated elements and B batches. C and R denote benchmark cases and repeats.

| Component | Technique/design | Time complexity | Key contract |
|---|---|---|---|
| `grad` / `jacobian` | Coordinate differences | `O(n²)` / `O(n² + mn)`, plus evaluations | Outputs (n,) / (m, n); space `O(n)` / `O(mn + n + m)`, excluding callable allocations |
| `Pipeline` | Sequential composition | Step costs + `O(ΣNᵢ)` | Extra array storage `O(max Nᵢ)`, excluding step internals/outputs; array-target copying adds `O(y.size)` |
| `FeatureUnion` | Branch concatenation | Branch costs + `O(bN + M)` for transformation | Array storage `O(N + M)`, excluding branch internals |
| Numerical utilities | Vectorised operations | Elementwise `O(N)`; reductions `O(N + Q)` | Float64 outputs; space `O(N + Q)` for reductions; equal-shaped distance inputs |
| `iter_batches` | Consecutive slices | `O(B)` | Auxiliary space `O(1)` at fixed rank, excluding retained batches |
| Benchmark harness | Repeated measurements | Proportional to calls and workload cost, plus checks | Report storage `O(CR)`, excluding workloads |

### Testing and limitations

The complete integrated suite passed **642 tests**, with four optional-PyYAML warnings. Tests cover analytic derivatives across all schemes, rectangular Jacobians, invalid arguments/outputs, ineffective steps and overflow; composition order, target forwarding, prediction, nesting, changed widths, copy isolation and failed refits; and utility axes, empty arrays, non-contiguous inputs, NaNs, infinities, extreme values, large signed/unsigned integers and batch views. Harness tests cover execution counts/order, correctness failure, configuration validation and mocked-clock arithmetic without performance thresholds. The independent review of the statistics and metrics fixes also reproduced the edge cases before integration.

Finite differences remain sensitive to step size and roundoff; callables must not mutate evaluation inputs. Results remain limited by float64 range. Object-array copies are shallow; steps control dtype, missing-value and feature policies. Compatibility with actual scalers, imputers and encoders requires integration testing.

The integrated benchmark set covers utilities, differentiation, pipeline composition, statistics and metrics. Utilities, statistics and metrics use correctness-checked loop or pairwise baselines; derivatives use untimed analytic checks. Composition uses direct utility comparisons and adapters that learn no parameters, measuring overhead rather than preprocessing training. Timings concern one environment and finite float64 workloads; preprocessing, sorting/searching, ranking and I/O do not currently have benchmark cases. The integrated sample-CSV pipeline smoke test and Quickstart script, notebook checker and fresh AIML-kernel notebook execution passed.
