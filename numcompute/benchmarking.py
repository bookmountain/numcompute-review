"""Correctness-checked microbenchmarks using only NumPy and the standard library."""

from dataclasses import dataclass, field
import platform
import sys
import time
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class BenchmarkCase:
    """Describe a prepared workload; setup is excluded from timing.

    Parameters
    ----------
    name : str
        Unique workload name.
    operation : callable
        Zero-argument operation returning a fully evaluated result. Callers
        must consume generators and reset mutable state within this callable
        if needed. Calls must be repeatable without changing the workload.
    check : callable
        Accepts the operation result and raises on incorrect values or shapes.
        Runs before timing; its cost is excluded.
    baseline : callable or None
        Optional zero-argument comparison performing equivalent work. Its
        output is checked with the same check before timing.
    baseline_label : str
        Description of the comparison (do not imply all comparisons are loops).
    metadata : dict
        JSON-serializable shapes, dtypes, parameters and workload notes.

    Notes
    -----
    Construction uses O(1) time/space, retaining supplied objects. There is no
    module-specific dispatch: future modules supply additional cases. Callable
    errors propagate when run; non-callable operations/checks raise TypeError.
    """

    name: str
    operation: Callable[[], Any]
    check: Callable[[Any], None]
    baseline: Callable[[], Any] | None = None
    baseline_label: str = "python_loop"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        if not callable(self.operation) or not callable(self.check):
            raise TypeError("operation and check must be callable")
        if self.baseline is not None and not callable(self.baseline):
            raise TypeError("baseline must be callable or None")


def _count(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _sample(operation, number):
    start = time.perf_counter_ns()
    for _ in range(number):
        operation()
    return (time.perf_counter_ns() - start) / (1e9 * number)


def _summary(samples):
    return {
        "samples_seconds_per_call": samples,
        "median_seconds": float(np.median(samples)),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def run_benchmarks(cases, *, repeats=5, number=3, warmups=1, seed=2026):
    """Check prepared cases and return a JSON-serializable timing report.

    Parameters
    ----------
    cases : iterable of BenchmarkCase
        Already prepared, repeatable workloads; names must be unique.
    repeats, number : int >= 1
        Number of samples and calls per sample, respectively.
    warmups : int >= 0
        Untimed calls per implementation after correctness validation.
    seed : int >= 0
        Recorded input-generation seed; this function does not generate data.

    Returns
    -------
    dict
        Environment/configuration and per-case timings in seconds per call.
        Ratios are baseline median / operation median, never assertions.

    Raises
    ------
    TypeError, ValueError
        Invalid counts, seed, case types or duplicate names.
    Exception
        Propagates workload/check errors; failed correctness aborts the run.

    Notes
    -----
    Both outputs are checked before timing. Sample order alternates to reduce
    order bias. GC retains its normal enabled/disabled state. Includes callable
    overhead and output allocation/disposal, excludes input setup and checks.
    For C cases, R repeats, N calls/sample and W warmups, time is
    O(C*(1+W+R*N)*T), where T bounds workload/check cost. Report space is O(C*R)
    plus workload storage. No timing-based pass/fail threshold is used.
    """
    for value, name, minimum in ((repeats, "repeats", 1), (number, "number", 1),
                                 (warmups, "warmups", 0), (seed, "seed", 0)):
        _count(value, name, minimum)
    cases = list(cases)
    if any(not isinstance(case, BenchmarkCase) for case in cases):
        raise TypeError("cases must contain BenchmarkCase objects")
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("case names must be unique")
    import contextlib
    import gc
    import io
    import os
    from datetime import datetime, timezone

    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    report = {
        "schema_version": 1,
        "environment": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version, "executable": sys.executable,
            "numpy": np.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "numpy_config": config.getvalue(),
            "thread_environment": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "gc_enabled": gc.isenabled(),
            "timer": "perf_counter_ns",
            "timer_resolution_seconds": time.get_clock_info("perf_counter").resolution,
        },
        "config": {"seed": seed, "repeats": repeats, "number": number,
                   "warmups": warmups, "sample_order": "alternating"},
        "results": [],
    }
    for case in cases:
        operations = {"operation": case.operation}
        if case.baseline is not None:
            operations["baseline"] = case.baseline
        for operation in operations.values():
            case.check(operation())
            for _ in range(warmups):
                operation()
        samples = {key: [] for key in operations}
        for repeat in range(repeats):
            order = list(operations)
            if repeat % 2:
                order.reverse()
            for key in order:
                samples[key].append(_sample(operations[key], number))
        result = {"name": case.name, "metadata": case.metadata,
                  "correctness": "passed", "operation": _summary(samples["operation"])}
        if case.baseline is not None:
            result["baseline_label"] = case.baseline_label
            result["baseline"] = _summary(samples["baseline"])
            elapsed = result["operation"]["median_seconds"]
            result["baseline_over_operation"] = (
                result["baseline"]["median_seconds"] / elapsed if elapsed else None
            )
        report["results"].append(result)
    return report
