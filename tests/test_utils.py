"""Numerical, validation, and input-preservation contracts for utilities."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from numcompute.utils import (
    euclidean_distance, iter_batches, logsumexp, manhattan_distance,
    relu, sigmoid, softmax,
)


DISTANCES = (euclidean_distance, manhattan_distance)
NUMERICAL = (sigmoid, relu, softmax, logsumexp, *DISTANCES)
AXIS_FUNCTIONS = (softmax, logsumexp, *DISTANCES)


def call_numeric(function, x, **kwargs):
    if function in DISTANCES:
        return function(x, np.zeros_like(x), **kwargs)
    return function(x, **kwargs)


def assert_float_output(result, expected):
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == np.shape(expected)
    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("function", NUMERICAL)
@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float32, np.float64])
def test_numeric_functions_preserve_noncontiguous_input(function, dtype):
    backing = np.array([1, 99, 2, 99, 3, 99], dtype=dtype)
    original = backing.copy()
    result = call_numeric(function, backing[::2])
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(backing, original)
    assert backing.dtype == original.dtype


@pytest.mark.parametrize("function", NUMERICAL)
@pytest.mark.parametrize("x", [
    [1, 2], np.array([True]), np.array([1j]),
    np.array([1], dtype=object), np.array(["1"]),
], ids=["list", "boolean", "complex", "object", "string"])
def test_numeric_functions_reject_invalid_input_type(function, x):
    with pytest.raises(TypeError, match="real numeric NumPy array"):
        call_numeric(function, x)


@pytest.mark.parametrize("function", AXIS_FUNCTIONS)
@pytest.mark.parametrize("axis, error", [
    (True, TypeError), (np.bool_(False), TypeError), (1.0, TypeError),
    ("0", TypeError), ((0,), TypeError), (2, ValueError), (-3, ValueError),
])
def test_invalid_axis(function, axis, error):
    with pytest.raises(error, match="axis"):
        call_numeric(function, np.ones((2, 3)), axis=axis)


@pytest.mark.parametrize("function", [softmax, *DISTANCES])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_finite_only_functions_reject_nonfinite_values(function, value):
    with pytest.raises(ValueError, match="finite"):
        call_numeric(function, np.array([1., value]))


@pytest.mark.parametrize("function, expected", [
    (sigmoid, [[1 / (1 + np.exp(2)), .5, 1 / (1 + np.exp(-2))]]),
    (relu, [[0., 0., 2.]]),
])
def test_activation_numerical_values(function, expected):
    assert_float_output(function(np.array([[-2., 0., 2.]])), expected)


@pytest.mark.parametrize("function, scalar", [(sigmoid, .5), (relu, 0.)])
def test_activation_scalar_and_empty_shapes(function, scalar):
    assert_float_output(function(np.array(0)), scalar)
    assert_float_output(function(np.empty((2, 0, 3))), np.empty((2, 0, 3)))


@pytest.mark.parametrize("function, expected", [
    (sigmoid, [0., 0., .5, 1., 1., np.nan]),
    (relu, [0., 0., 0., 1e308, np.inf, np.nan]),
])
def test_activation_extremes_and_nan(function, expected):
    with np.errstate(all="raise"):
        result = function(np.array([-np.inf, -1e308, 0., 1e308, np.inf, np.nan]))
    assert_float_output(result, expected)


@pytest.mark.parametrize("axis", [0, 1, -1, -2, np.int64(0)])
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_softmax_axis_normalization_and_shape(axis, dtype):
    x = np.array([[0, 1, 2], [2, 0, 1]], dtype=dtype)
    weights = np.exp(x)
    expected = weights / weights.sum(axis=axis, keepdims=True)
    result = softmax(x, axis=axis)
    assert_float_output(result, expected)
    np.testing.assert_allclose(result.sum(axis=axis), 1.)
    assert np.all((result >= 0) & (result <= 1))


def test_softmax_large_finite_logits():
    with np.errstate(all="raise"):
        assert_float_output(softmax(np.array([1000., 1001.])),
                            np.array([1., np.e]) / (1 + np.e))
        assert_float_output(softmax(np.array([-1e308, 1e308])), [0., 1.])
        assert_float_output(softmax(np.array([1e308, 1e308])), [.5, .5])


def test_softmax_empty_and_scalar_shapes():
    assert_float_output(softmax(np.empty((0, 3))), np.empty((0, 3)))
    for x in (np.array(1.), np.empty((2, 0))):
        with pytest.raises(ValueError):
            softmax(x)


@pytest.mark.parametrize("axis", [None, 0, 1, -1, -2, np.int64(0)])
@pytest.mark.parametrize("keepdims", [False, True, np.bool_(True)])
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_logsumexp_reduction_shape(axis, keepdims, dtype):
    x = np.array([[0, 1, 2], [2, 0, 1]], dtype=dtype)
    expected = np.log(np.sum(np.exp(x), axis=axis, keepdims=bool(keepdims)))
    assert_float_output(logsumexp(x, axis=axis, keepdims=keepdims), expected)


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_logsumexp_scalar_and_empty(dtype):
    assert_float_output(logsumexp(np.array(3, dtype=dtype)), 3.)
    assert_float_output(logsumexp(np.empty((2, 0), dtype=dtype), axis=1),
                        [-np.inf, -np.inf])
    assert_float_output(logsumexp(np.empty((0, 3), dtype=dtype), axis=1), [])
    assert_float_output(logsumexp(np.array([], dtype=dtype)), -np.inf)
    with pytest.raises(ValueError, match="axis"):
        logsumexp(np.array(3), axis=0)


@pytest.mark.parametrize("keepdims", [0, 1, None, "yes"])
def test_logsumexp_invalid_keepdims(keepdims):
    with pytest.raises(TypeError, match="keepdims"):
        logsumexp(np.ones(2), keepdims=keepdims)


def test_logsumexp_nonfinite_slices_and_large_finite_values():
    x = np.array([[-np.inf, -np.inf], [np.inf, 0.], [np.nan, np.inf],
                  [1000., 1000.], [-1e308, 1e308]])
    with np.errstate(all="raise"):
        result = logsumexp(x, axis=1)
    assert_float_output(result, [-np.inf, np.inf, np.nan, 1000 + np.log(2), 1e308])


@pytest.mark.parametrize("function", DISTANCES)
@pytest.mark.parametrize("axis", [0, 1, -1, -2, np.int64(0)])
def test_distance_values_axes_and_both_inputs_preserved(function, axis):
    x = np.array([[1., 4.], [5., 8.], [9., 12.]]).T
    y = np.array([[1, 1], [1, 5], [6, 8]]).T
    originals = (x.copy(), y.copy())
    difference = x - y
    expected = (np.sqrt(np.sum(difference**2, axis=axis))
                if function is euclidean_distance
                else np.sum(np.abs(difference), axis=axis))
    assert_float_output(function(x, y, axis=axis), expected)
    np.testing.assert_array_equal(x, originals[0])
    np.testing.assert_array_equal(y, originals[1])


@pytest.mark.parametrize("function", DISTANCES)
def test_distance_scalar_output_empty_axes_and_symmetry(function):
    x, y = np.array([1., 4.]), np.array([4., 0.])
    assert_float_output(function(x, y), 5. if function is euclidean_distance else 7.)
    np.testing.assert_array_equal(function(x, y), function(y, x))
    assert_float_output(function(x, x), 0.)
    assert_float_output(function(np.empty((2, 0)), np.empty((2, 0))), [0., 0.])
    assert_float_output(function(np.empty((0, 3)), np.empty((0, 3))), [])


@pytest.mark.parametrize("function", DISTANCES)
@pytest.mark.parametrize("x, y, error", [
    (np.array(1), np.array(2), ValueError),
    (np.ones((2, 3)), np.ones(3), ValueError),
    (np.ones(2), [1, 2], TypeError),
    (np.ones(2), np.array([True, False]), TypeError),
    (np.ones(2), np.array([1j, 2j]), TypeError),
    (np.ones(2), np.array([1, 2], dtype=object), TypeError),
    (np.ones(2), np.array([np.nan, 0.]), ValueError),
    (np.ones(2), np.array([np.inf, 0.]), ValueError),
])
def test_distance_invalid_shapes_and_second_input(function, x, y, error):
    with pytest.raises(error):
        function(x, y)


@pytest.mark.parametrize("function", DISTANCES)
@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_distance_stability_without_unnecessary_overflow_or_underflow(function, scale):
    with np.errstate(all="raise"):
        result = function(np.array([3*scale, 4*scale]), np.zeros(2))
    assert_float_output(result, (5 if function is euclidean_distance else 7)*scale)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("base", [2**53 - 1, 2**53, 2**53 + 1])
def test_large_integer_gaps_are_preserved(dtype, base):
    x = np.array([base, base + 1], dtype=dtype)
    y = np.array([base + 3, base + 5], dtype=dtype)
    assert_float_output(euclidean_distance(x, y), 5.)
    assert_float_output(manhattan_distance(x, y), 7.)
    assert_float_output(softmax(x), np.array([1., np.e]) / (1 + np.e))


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("base", [2**53 - 1, 2**53 + 1, 2**53 + 3])
@pytest.mark.parametrize("gap", [0, 1])
def test_logsumexp_large_integer_rounding_residual(dtype, base, gap):
    # Decimal supplies an independent oracle; relative tolerance at this
    # magnitude would hide the one-ULP regression this test guards against.
    with localcontext() as context:
        context.prec = 80
        expected = float(Decimal(base) + (1 + Decimal(-gap).exp()).ln())
    result = logsumexp(np.array([base, base - gap], dtype=dtype))
    assert result.shape == ()
    assert result.dtype == np.float64
    assert result.item() == expected


@pytest.mark.parametrize("function", DISTANCES)
@pytest.mark.parametrize("left, right, left_dtype, right_dtype", [
    (-2**63, 2**63 - 1, np.int64, np.int64),
    (0, 2**64 - 1, np.uint64, np.uint64),
    (-2**63, 2**64 - 1, np.int64, np.uint64),
])
def test_integer_distance_subtraction_does_not_wrap(function, left, right,
                                                   left_dtype, right_dtype):
    x, y = np.array([left], dtype=left_dtype), np.array([right], dtype=right_dtype)
    assert_float_output(function(x, y), float(abs(left - right)))
    assert_float_output(function(y, x), float(abs(left - right)))


@pytest.mark.parametrize("drop_last", [False, True, np.bool_(True)])
@pytest.mark.parametrize("batch_size", [1, 2, 5, 8, np.int64(2)])
@pytest.mark.parametrize("dtype", [np.int64, np.float64, object, str, bool])
def test_batches_order_shapes_dtype_and_input_preservation(drop_last, batch_size, dtype):
    x = np.arange(30).reshape(5, 2, 3).astype(dtype)
    original = x.copy()
    batches = list(iter_batches(x, batch_size, drop_last=drop_last))
    count = len(x) // batch_size if drop_last else (len(x) + batch_size - 1) // batch_size
    assert len(batches) == count
    for index, batch in enumerate(batches):
        expected = x[index*batch_size:min((index + 1)*batch_size, len(x))]
        assert batch.shape == expected.shape
        assert batch.dtype == x.dtype
        assert np.shares_memory(batch, x)
        np.testing.assert_array_equal(batch, expected)
    np.testing.assert_array_equal(x, original)


def test_batches_noncontiguous_views_mutate_backing_array():
    backing = np.arange(12)
    x = backing[::2]
    batches = list(iter_batches(x, 2))
    np.testing.assert_array_equal(np.concatenate(batches), x)
    batches[1][0] = -1
    assert x[2] == backing[4] == -1
    backing[6] = -2
    assert batches[1][1] == -2


@pytest.mark.parametrize("drop_last", [False, True])
def test_batches_empty_input(drop_last):
    assert list(iter_batches(np.empty((0, 3)), 2, drop_last=drop_last)) == []


@pytest.mark.parametrize("x, batch_size, drop_last, error", [
    ([1, 2], 1, False, TypeError),
    (np.array(1), 1, False, ValueError),
    (np.ones(2), 0, False, ValueError),
    (np.ones(2), -1, False, ValueError),
    (np.ones(2), True, False, TypeError),
    (np.ones(2), np.bool_(True), False, TypeError),
    (np.ones(2), 1.0, False, TypeError),
    (np.ones(2), "1", False, TypeError),
    (np.ones(2), 1, 1, TypeError),
    (np.ones(2), 1, None, TypeError),
])
def test_batches_invalid_arguments(x, batch_size, drop_last, error):
    # Consume the iterator because validation is allowed to be lazy.
    with pytest.raises(error):
        list(iter_batches(x, batch_size, drop_last=drop_last))
