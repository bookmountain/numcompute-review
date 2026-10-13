## Sorting, Selection, Searching and Ranking — Shushan

### Design and implementation

Shushan's component implements reusable sorting, partial-selection,
searching, ranking, and percentile operations in `sort_search.py` and `rank.py`.
Functions accept NumPy-compatible numeric input, return new arrays or scalar
results, and do not mutate caller data. Shared validation gives consistent
errors for unsupported dtypes, shapes, non-finite values, and invalid integer
parameters. Index conventions are explicit: `topk` treats `k` as a count,
whereas `quickselect` treats it as a zero-based position.

`stable_sort` delegates to `numpy.sort(kind="stable")`, preserving the input
order of equal values. `multi_key_sort` uses `numpy.lexsort`; requested columns
are reversed because NumPy treats its final key as primary. Both use compiled
NumPy ordering rather than Python element loops.

`topk` uses `numpy.argpartition` to find a boundary without sorting the complete
input. Values strictly better than the boundary are retained, remaining places
are filled from the earliest tied indexes, and only the selected subset is
ordered. This produces deterministic ties and expected `O(n + k log k)` time,
compared with `O(n log n)` for a full sort. Integer values are not converted to
floating point, preserving large unsigned inputs.

`quickselect` copies the input, chooses a pivot, creates vectorised
below/equal/above partitions, and continues only in the partition containing
the requested position. It has expected `O(n)` and worst-case `O(n²)` time. Its
control loop is necessary because each partition depends on the previous one.
`binary_search` similarly uses an algorithmic loop to halve an already sorted
interval. It returns the leftmost insertion position and an existence flag in
`O(log n)` time. It does not scan for sortedness because that would make the
operation `O(n)`.

Ranking stably sorts valid observations, identifies equal-value groups,
assigns average, dense, or ordinal one-based ranks, and maps results back to
their original positions. Percentiles support linear, lower, higher, and
midpoint interpolation, axis-based calculation, and explicit NaN policies.

| Operation | Technique | Time complexity | Key contract |
|---|---|---:|---|
| Stable/multi-key sort | Stable sort/`lexsort` | `O(n log n)` | Stable ties; first listed key is primary |
| Top-k | Partial partition | `O(n + k log k)` | Ties prefer earlier indexes |
| Quickselect | Three-way partitioning | Expected `O(n)` | `k` is zero-based |
| Binary search | Interval halving | `O(log n)` | Input is already sorted |
| Ranking/percentile | Ordering and interpolation | `O(n log n)` upper bound | Explicit tie and NaN semantics |

### Testing and limitations

Thirteen unit tests use hand-computed expectations for multi-key priority,
every quickselect position, leftmost binary-search results, all ranking methods,
and percentile interpolation. Edge cases cover empty arrays, `k=0`, invalid and
extreme `k`, all-equal values, boundary ties, duplicate search values, NaNs,
non-contiguous views, large `uint64` values, invalid options, and input
non-mutation. The independent branch suite currently passes all tests.

Binary search assumes ascending finite data, while `topk` and `quickselect`
reject NaNs because their selection order would be ambiguous. Ranking and
percentiles instead provide omit, propagate, and raise policies. In the final
quickstart, a numeric CSV column will be supplied by the I/O component; missing
values must be filtered or imputed before finite-only selection and search.
Keeping parsing separate from the algorithm layer makes these operations
reusable with any compatible numeric dataset.
