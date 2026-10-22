"""Harness semantics, without timing thresholds."""
import unittest
from unittest.mock import patch

from numcompute.benchmarking import BenchmarkCase, run_benchmarks


class BenchmarkTests(unittest.TestCase):
    def test_timing_arithmetic_with_mocked_clock(self):
        case = BenchmarkCase("clock", lambda: 7, lambda value: None, lambda: 7)
        # Four calls per sample: operation totals 1 s then 3 s;
        # baseline totals 4 s then 8 s (second repeat reverses order).
        ticks = [10_000_000_000, 11_000_000_000,
                 12_000_000_000, 16_000_000_000,
                 20_000_000_000, 28_000_000_000,
                 30_000_000_000, 33_000_000_000]
        with patch("numcompute.benchmarking.time.perf_counter_ns", side_effect=ticks) as clock:
            report = run_benchmarks([case], repeats=2, number=4, warmups=1)
        self.assertEqual(clock.call_count, 8)
        row = report["results"][0]
        self.assertEqual(row["operation"], {
            "samples_seconds_per_call": [0.25, 0.75],
            "median_seconds": 0.5, "min_seconds": 0.25, "max_seconds": 0.75,
        })
        self.assertEqual(row["baseline"], {
            "samples_seconds_per_call": [1.0, 2.0],
            "median_seconds": 1.5, "min_seconds": 1.0, "max_seconds": 2.0,
        })
        self.assertEqual(row["baseline_over_operation"], 3.0)

    def test_checks_both_implementations_and_consumes_expected_calls(self):
        calls = []
        checked = []
        case = BenchmarkCase("example", lambda: calls.append("op") or 7,
                             checked.append, lambda: calls.append("base") or 7)
        report = run_benchmarks([case], repeats=2, number=3, warmups=1)
        self.assertEqual(checked, [7, 7])
        self.assertEqual(calls.count("op"), 8)
        self.assertEqual(calls.count("base"), 8)
        # Timed blocks alternate which implementation runs first.
        self.assertEqual(calls[4:], ["op"] * 3 + ["base"] * 6 + ["op"] * 3)
        row = report["results"][0]
        self.assertEqual(len(row["operation"]["samples_seconds_per_call"]), 2)

    def test_incorrect_baseline_aborts_before_timing(self):
        calls = []
        def check(value):
            self.assertEqual(value, 1)
        case = BenchmarkCase("bad", lambda: calls.append(1) or 1, check, lambda: 2)
        with self.assertRaises(AssertionError):
            run_benchmarks([case], warmups=0)
        self.assertEqual(calls, [1])

    def test_no_baseline_and_invalid_configuration(self):
        case = BenchmarkCase("solo", lambda: 1, lambda value: None)
        report = run_benchmarks([case], repeats=1, number=1, warmups=0)
        self.assertNotIn("baseline", report["results"][0])
        for options in ({"number": 0}, {"repeats": 0}, {"warmups": -1}, {"seed": -1}):
            with self.assertRaises(ValueError):
                run_benchmarks([case], **options)
        with self.assertRaises(TypeError):
            run_benchmarks([case], repeats=True)
        with self.assertRaises(ValueError):
            run_benchmarks([case, case])


if __name__ == "__main__":
    unittest.main()
