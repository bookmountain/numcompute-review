# Contribution record

Record the work each member actually completes. Assigned ownership by itself is
not evidence of contribution.

| Member | Modules reviewed/changed | New tests or issues found | Review of another member's work | Demo/report/video contribution | Evidence (commit/PR/date) |
|---|---|---|---|---|---|
| Katherine Roman (A) | Implemented CSV loading, chunk iteration, and four preprocessing classes; documented installation in README.md. | 41 CSV and preprocessing tests passed locally. | Pending | Pending | Pending |
| Shushan | Pending | Pending | Pending | Pending | Pending |
| Lachlan | Implemented `stats.py` (axis-aware mean/median/var/std/min/max, histogram, quantile, mergeable Welford/Chan `RunningStats` with NaN policies) and `metrics.py` (confusion matrix, accuracy, precision/recall/F1 with multiclass averaging, MSE, ROC curve, AUC), Added seven correctness-checked stats/metrics benchmark cases to `benchmark/run.py` and regenerated `benchmark/results.json`. AI-assisted implementation, reviewed and tested locally. | 30 new tests in `tests/test_stats_metrics.py` (axes/shapes, all-NaN slices, streaming merges, undefined ratios, averaging, score ties, invalid inputs); full suite 563 passed. | Pending | Pending | PR `stats-metrics` → `main`, 2026-10-24 |
| D: name pending | Pending | Pending | Pending | Pending | Pending |

## Decisions and integration log

| Date | Decision or issue | Owner | Resolution/evidence |
|---|---|---|---|
| Pending | Pending group review | Pending | Pending |
| 2026-10-22 | Created the reusable benchmark harness and runner for currently available utilities, optimisation and pipeline composition; added usage documentation and three harness tests. | AI-assisted work requested by repository user; member attribution/review pending | `benchmark/results.json`: one default run, 13 correctness-checked workloads; three harness tests passed. Existing numerical tests could not run because pytest is unavailable. Uncommitted; no PR review claimed. |

Record only work that was actually completed. Useful evidence includes commit
hashes, pull-request links, review links, test output dates, benchmark result
files, report sections, and video timestamps. Assigned ownership by itself is
not a completed contribution.

Benchmark follow-up (2026-10-22, AI-assisted; member attribution/review pending): added untimed exact batch-content/order and input-preservation checks, plus a deterministic mocked-clock arithmetic test. Complete suite passed using `/private/tmp/numcompute-pytest-deps`: 478 tests. Regenerated `benchmark/results.json`; all 13 workloads passed and all six recorded source hashes matched. No numerical implementations changed; no commit or push.
