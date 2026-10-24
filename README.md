# NumCompute group assignment

NumCompute is a four-person Python and NumPy scientific-computing toolkit. This
repository starts with a shared package scaffold on `main`; each member develops
and tests only their allocated area on a separate branch. Reviewed pull requests
are then integrated into the final package, notebook, report, and video.

Read [GROUP_ALLOCATION.md](GROUP_ALLOCATION.md) before starting. It defines the
files, demonstrations, reviews, and completion criteria owned by every member.
The saved course-page printout and core specification are mapped in
[ASSIGNMENT_REQUIREMENTS.md](ASSIGNMENT_REQUIREMENTS.md).
Actual completed work must be recorded in
[CONTRIBUTIONS.md](CONTRIBUTIONS.md).

## Course-page requirements snapshot

The locally saved printout of the university course page displays
**26 October 2026** and states that submission is due at **9:00 am on the Monday
after the mid-semester break**. Verify the live course page before submission.
The printout describes group work for 3–4 students, worth 30%, using only plain
Python and NumPy—no pandas, scikit-learn, PyTorch, or other external data/ML
libraries.

The group submission requires a Git repository, at least 20 meaningful unit
tests, an end-to-end `demo/quickstart.ipynb`, reproducible benchmarks, a video of
at most five minutes, and a PDF report of at most four pages.

## Installation and preprocessing checks

Python 3.10 or newer is required. NumPy 1.24 or newer is the
only runtime dependency.

Download and extract the repository ZIP, or clone the repository.
Open a terminal in the project folder containing `pyproject.toml`.

Install the package and its dependency:

```sh
python -m pip install -e .
```

The editable installation uses the code in this project folder,
so saved code changes are available without reinstalling.

Run the CSV and preprocessing tests:

```sh
python -m unittest discover -s tests -p test_io_preprocessing.py -v
```

Run the preprocessing demonstration:

```sh
python -m demo.preprocessing_demo
```

The demonstration uses `demo/data/sample.csv` and prints the
output shapes. It fits the preprocessing tools on the training
rows and reuses those fitted values for the test rows.

## Branch ownership

| Member | Area | Branch |
|---|---|---|
| A: pending | CSV I/O and preprocessing | `data-preprocessing` |
| Shushan | Sorting, searching, ranking, and percentiles | `book/algorithm` |
| C: pending | Statistics, metrics, and benchmarking | `stats-metrics` |
| Ruofan| Pipelines, optimisation, utilities, and integration | `pipeline-optim` |

Members A, C, and D should replace `pending` with their names and GitHub
usernames. Branch ownership includes implementation, tests, documentation,
explanation, and review—not merely creating files.

## Required shared outputs

The repository is not complete when the module code passes. The group must also
produce:

- `demo/data/sample.csv`, supplied by A, with numeric values, ties, categories,
  and missing data suitable for the demonstrations;
- `demo/quickstart.py` and `demo/quickstart.ipynb`, assembled by D from examples
  supplied by all four members;
- `demo/check_notebook.py`, maintained by D, to execute notebook code in order;
- reproducible benchmark code and recorded results, owned by C;
- a technical report within the assignment page limit, assembled from each
  member's written section;
- a demonstration video within the assignment time limit, with every member
  explaining their own work;
- an accurate contribution and review record in `CONTRIBUTIONS.md`.

## Quickstart

From the repository root, run the script and execute the notebook checker:

```sh
python3 -B demo/quickstart.py
python3 -B demo/check_notebook.py
```

To run the notebook interactively, open the repository folder in VS Code, open
`demo/quickstart.ipynb`, select the existing **Python (AIML Course)** kernel,
and choose **Run All**. The `jupyter notebook` front end is unavailable in the
current course environment.

## Quickstart ownership

D coordinates the quickstart, but does not write everyone else's content:

- A supplies CSV loading and preprocessing examples plus the sample dataset.
- Shushan supplies sorting, top-k, quickselect, binary-search, ranking, and percentile
  examples with expected outputs.
- C supplies statistics, evaluation-metric, and benchmark examples.
- D supplies pipeline, gradient/Jacobian, and utility examples, combines all
  sections, and verifies the script and notebook from a fresh environment.

## Branch workflow

```mermaid
flowchart LR
    M[main: shared scaffold] --> A[A: data-preprocessing]
    M --> B[Shushan: book/algorithm]
    M --> C[C: stats-metrics]
    M --> D[D: pipeline-optim]
    A --> RA[A reviews Shushan]
    B --> RB[Shushan reviews C]
    C --> RC[C reviews D]
    D --> RD[D reviews A]
    RA --> I[reviewed main]
    RB --> I
    RC --> I
    RD --> I
    I --> Q[D assembles quickstart and package integration]
    Q --> V[all members verify tests, notebook, report, and video]
    V --> S[nominated member submits]
```

Create an assigned branch from `main`, never from another member's unfinished
branch:

```sh
git clone https://github.com/bookmountain/numcompute-review.git
cd numcompute-review
git switch -c <assigned-branch> origin/main
git push -u origin <assigned-branch>
```

Commit with your own Git identity. Before requesting review, run your scoped
tests, document important edge cases and complexity, and add at least one
meaningful test. Open a pull request into `main`; the reviewer named in the
allocation must approve it before integration.

## Definition of done

A member's allocation is complete only when:

1. Their required functions and error behaviour are implemented.
2. Their scoped tests pass and cover ordinary and edge cases.
3. They can explain the algorithms, shapes, numerical choices, and limitations.
4. They have supplied their quickstart, report, and video material.
5. Their assigned cross-review is recorded with a commit or pull-request link.
6. Their actual contribution row is completed in `CONTRIBUTIONS.md`.

After all module pull requests are merged, D creates the integration branch,
assembles the quickstart, runs the complete test suite and notebook, and opens
the final integration pull request. The entire group reviews the final result.

## Current benchmarks

From the repository root (Python 3.10+ and NumPy installed):

```sh
python3 benchmark/run.py
python3 benchmark/run.py --rows 4000 --features 32 --repeats 7 --number 5 --seed 2026 --output benchmark/results-large.json
```

The reusable harness is `numcompute/benchmarking.py`; current workload definitions
and the CLI are in `benchmark/run.py`. The default run writes
`benchmark/results.json` and prints a median-time table. JSON retains all samples
in seconds per call, min/median/max, correctness status, seed, shapes, dtypes,
tolerances, repetitions, warmups, Python/NumPy/platform/CPU and NumPy build details,
thread environment, source hashes, and available Git provenance. Input setup and
correctness checks are outside timing. Each sample averages `--number` calls;
implementation order alternates across samples. GC is left in its existing state.

Current coverage uses only implemented modules:

- `utils`: sigmoid, ReLU, softmax, logsumexp, Euclidean/Manhattan distances and
  fully consumed batching, compared with explicit Python loops.
- `optim`: central finite-difference gradient and Jacobian of quadratic functions,
  checked against analytic derivatives; no timed analytic baseline.
- `pipeline`: Pipeline and FeatureUnion transform/fit_transform, using benchmark
  adapters around existing ReLU/sigmoid utilities. Direct utility calls are the
  comparison. Fitting these adapters is a no-op; fit_transform includes fresh
  construction, while transform uses pre-fitted objects. These ratios measure
  composition overhead, not a loop-versus-vectorisation speedup. Estimator
  prediction is not covered because no concrete estimator is implemented.
- `stats`: column-wise mean and sample variance against explicit two-pass loops,
  one batched `RunningStats.update` against a per-element Welford loop, and a
  32-bin histogram against an edge-scanning loop.
- `metrics`: confusion matrix and macro F1 on a seeded four-class problem
  against pure-Python counting (small ratios: at this size both are dominated
  by call overhead), and ROC AUC with tied scores checked against the O(P·N)
  pairwise Mann–Whitney count.

Both timed implementations must pass shape/value checks before measurement.
The runner uses finite standard-normal float64 inputs; these measurements are
not exhaustive numerical edge-case tests. Ratios are baseline median divided by
operation median and are environment-specific evidence, never test thresholds.
Small workloads include Python-call and allocation overhead; CPU load and thread
settings can affect results. A fixed seed reproduces inputs, not elapsed times.

To extend coverage once another module is implemented, add a case factory using
its actual API and append its `BenchmarkCase` objects in `available_cases`.
Each case provides a zero-argument operation, a correctness checker, optional
equivalent baseline and descriptive metadata. Consume lazy results inside the
operation; reset mutable state there, or prepare a safely reusable fitted object
outside timing. No harness change is needed for preprocessing, sorting/searching,
ranking or I/O; no cases or imports for those modules are supplied now.

Run the harness tests using only the standard library and NumPy:

```sh
python3 -m unittest discover -s tests -p test_benchmarking.py
```

The existing numerical test suite uses pytest and can be run with
`python3 -m pytest tests` when that development dependency is installed.
