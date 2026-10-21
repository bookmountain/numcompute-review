## Sorting, searching and ranking (Shushan)

### Design and implementation

My part of NumCompute is `sort_search.py` and `rank.py`. All functions take
NumPy-compatible numeric input, return new results without modifying the
caller's data, and use the shared `_validation.py` helpers so errors are
consistent across modules. One convention to watch: `k` is a count in `topk`
but a zero-based position in `quickselect`.

`stable_sort` wraps `numpy.sort(kind="stable")`, so equal values keep their
input order. `multi_key_sort` uses `numpy.lexsort`. Because `lexsort` treats
the last column it is given as the most important, the function reverses the
user's column list first.

`topk` avoids sorting the whole array. `numpy.argpartition` finds the k-th best
value, which acts as a cutoff. Every value better than the cutoff is kept, and
any places still free are filled with values equal to the cutoff, earliest
first, so ties always resolve the same way. Only these k values are then
sorted, giving O(n + k log k) time instead of O(n log n).

`quickselect` is a hand-written version of the same idea. It picks a pivot,
splits the values into those below, equal to and above it, and keeps only the
group that contains position k. It repeats this until position k falls on the
pivot. This takes expected O(n) time, with an O(n²) worst case. The repetition
needs a Python loop because each step uses the previous step's result, but the
splitting inside each step is vectorised.

`binary_search` repeatedly halves the search range of a sorted array and
returns the leftmost insertion index and whether the value exists, in O(log n)
time. It does not check whether the array is sorted or contains NaN or infinite
values, because that would mean reading every element and cost O(n). Providing valid input is left
to the caller, as the docstring states.

`rank` sorts the non-NaN values, groups values that are equal, and assigns
ordinal, dense or average ranks. I added `ascending=False` so that the highest
exam score can be rank 1. `percentile` validates its arguments and then calls
NumPy, supporting four interpolation methods, calculation along an axis, and
three NaN policies.

| Operation             | Technique                 |          Time complexity | Key contract                                       |
| --------------------- | ------------------------- | -----------------------: | -------------------------------------------------- |
| Stable/multi-key sort | Stable sort/`lexsort`     |             `O(n log n)` | Stable ties; first listed column is most important |
| Top-k                 | Partial partition         |         `O(n + k log k)` | Ties prefer earlier indexes                        |
| Quickselect           | Three-way partitioning    |          Expected `O(n)` | `k` is zero-based                                  |
| Binary search         | Interval halving          |               `O(log n)` | Input is already sorted and finite                 |
| Ranking/percentile    | Sorting and interpolation | `O(n log n)` upper bound | Explicit tie, direction and NaN rules              |

### Testing and limitations

Fourteen unit tests compare each function's output with answers worked out by
hand, including all ranking methods in both directions and every interpolation
method. Edge cases cover empty arrays, extreme `k`, all-equal values, ties at
the top-k cutoff, duplicates, NaNs, non-contiguous views, large `uint64` values,
boolean input, and checks that inputs are not modified. All tests pass.

NaN handling differs on purpose. `topk` and `quickselect` reject NaN because it
has no meaningful position in sorted order, `binary_search` expects sorted data
with no NaN or infinite values, and `rank` and `percentile` let the caller choose to omit,
propagate or raise. The quickstart therefore has to filter or impute missing
CSV values before using selection or search.
