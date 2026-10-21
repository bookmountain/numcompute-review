"""Tests for finite-difference gradients and Jacobians only."""

import numpy as np
import pytest

from numcompute.optim import grad, jacobian


METHODS = ("forward", "backward", "central")


def scalar_function(x):
    return x[0]**2 + 3*x[1]


def vector_function(x):
    return np.array([x[0]**2 + x[1], 3*x[0] - 2*x[1]])


@pytest.fixture(params=[(grad, scalar_function), (jacobian, vector_function)],
                ids=["grad", "jacobian"])
def derivative_case(request):
    return request.param


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [float, int], ids=["float", "integer"])
def test_grad_matches_analytical_result_and_preserves_input(method, dtype):
    x = np.array([2, 4], dtype=dtype)
    original = x.copy()

    result = grad(scalar_function, x, method=method, step=1e-5)

    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape
    # One-sided differences have an O(step) error for this quadratic.
    np.testing.assert_allclose(result, [4, 3], rtol=0, atol=2e-5)
    np.testing.assert_array_equal(x, original)
    assert x.dtype == original.dtype


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [float, int], ids=["float", "integer"])
def test_jacobian_matches_analytical_result_and_preserves_input(method, dtype):
    x = np.array([2, 4], dtype=dtype)
    original = x.copy()

    result = jacobian(vector_function, x, method=method, step=1e-5)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, [[4, 1], [3, -2]], rtol=0, atol=2e-5)
    np.testing.assert_array_equal(x, original)
    assert x.dtype == original.dtype


@pytest.mark.parametrize("method", METHODS)
def test_jacobian_rectangular_shape_has_outputs_as_rows(method):
    def f(x):
        return np.array([x[0] + x[1], 2*x[0] - x[1], 3*x[1]])

    result = jacobian(f, np.array([2., 4.]), method=method)

    assert result.shape == (3, 2)
    np.testing.assert_allclose(result, [[1, 1], [2, -1], [0, 3]],
                               rtol=0, atol=1e-8)


@pytest.mark.parametrize("method", METHODS)
def test_noncontiguous_input_and_backing_array_are_preserved(derivative_case, method):
    differentiate, f = derivative_case
    backing = np.array([2., 99., 4., 99.])
    original = backing.copy()
    x = backing[::2]

    result = differentiate(f, x, method=method)

    expected = [4, 3] if differentiate is grad else [[4, 1], [3, -2]]
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-5)
    np.testing.assert_array_equal(backing, original)


def test_non_callable_function_is_rejected(derivative_case):
    differentiate, _ = derivative_case
    with pytest.raises(TypeError, match="f must be callable"):
        differentiate(None, np.array([2., 4.]))


@pytest.mark.parametrize("method", ["unknown", "Central", None, 1])
def test_invalid_method_is_rejected(derivative_case, method):
    differentiate, f = derivative_case
    with pytest.raises(ValueError, match="method must be"):
        differentiate(f, np.array([2., 4.]), method=method)


@pytest.mark.parametrize("step", [0., -1., np.nan, np.inf, -np.inf])
def test_nonpositive_or_nonfinite_step_is_rejected(derivative_case, step):
    differentiate, f = derivative_case
    with pytest.raises(ValueError, match="step must be positive and finite"):
        differentiate(f, np.array([2., 4.]), step=step)


@pytest.mark.parametrize("step", [True, "small", 1j, [1e-5]])
def test_non_real_scalar_step_is_rejected(derivative_case, step):
    differentiate, f = derivative_case
    with pytest.raises(TypeError, match="step must be a real numeric scalar"):
        differentiate(f, np.array([2., 4.]), step=step)


@pytest.mark.parametrize("x", [np.array(2.), np.array([[2., 4.]]), np.array([])],
                         ids=["scalar", "matrix", "empty"])
def test_invalid_input_shape_is_rejected(derivative_case, x):
    differentiate, f = derivative_case
    with pytest.raises(ValueError, match="nonempty one-dimensional array"):
        differentiate(f, x)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_is_rejected(derivative_case, value):
    differentiate, f = derivative_case
    with pytest.raises(ValueError, match="x must contain only finite values"):
        differentiate(f, np.array([value, 4.]))


@pytest.mark.parametrize("x", [np.array(["2", "4"]), np.array([2j, 4j])],
                         ids=["string", "complex"])
def test_non_real_input_is_rejected(derivative_case, x):
    differentiate, f = derivative_case
    with pytest.raises(TypeError, match="x must contain real numeric values"):
        differentiate(f, x)


@pytest.mark.parametrize("output, error, message", [
    ([1., 2.], ValueError, "f must return a scalar"),
    (np.nan, ValueError, "f must return a finite value"),
    (np.inf, ValueError, "f must return a finite value"),
    ("invalid", TypeError, "f must return a real numeric scalar"),
    (1j, TypeError, "f must return a real numeric scalar"),
])
def test_grad_invalid_function_output_is_rejected(output, error, message):
    with pytest.raises(error, match=message):
        grad(lambda x: output, np.array([2., 4.]))


@pytest.mark.parametrize("output, error, message", [
    (1., ValueError, "nonempty one-dimensional array"),
    (np.array([]), ValueError, "nonempty one-dimensional array"),
    (np.array([[1., 2.]]), ValueError, "nonempty one-dimensional array"),
    (np.array([np.nan]), ValueError, "f must return only finite values"),
    (np.array([np.inf]), ValueError, "f must return only finite values"),
    (np.array(["invalid"]), TypeError, "f must return real numeric values"),
    (np.array([1j]), TypeError, "f must return real numeric values"),
])
def test_jacobian_invalid_function_output_is_rejected(output, error, message):
    with pytest.raises(error, match=message):
        jacobian(lambda x: output, np.array([2., 4.]))


@pytest.mark.parametrize("method", METHODS)
def test_jacobian_inconsistent_output_shape_is_rejected(method):
    def f(x):
        return np.array([1., 2.]) if x[0] == 2. else np.array([1.])

    with pytest.raises(ValueError, match="same output shape"):
        jacobian(f, np.array([2., 4.]), method=method)


@pytest.mark.parametrize("method", METHODS)
def test_step_too_small_to_change_coordinate_is_rejected(derivative_case, method):
    differentiate, f = derivative_case
    with pytest.raises(ValueError, match="step is too small"):
        differentiate(f, np.array([2., 4.]), method=method, step=1e-20)


@pytest.mark.parametrize("method, offset, step", [
    ("forward", -1., 2.),
    ("backward", 1., 2.),
    ("central", 0., 1.),
])
def test_nonfinite_derivative_raises_overflow_value_error(
    derivative_case, method, offset, step
):
    differentiate, _ = derivative_case

    def f(x):
        # Each sampled value is finite, but subtracting +/-1e308 overflows.
        value = 1e308 * (x[0] + offset)
        assert np.isfinite(value)
        return float(value) if differentiate is grad else np.array([value])

    x = np.array([0.])
    original = x.copy()
    # Ignore expected arithmetic warnings while checking the explicit error.
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="derivative is non-finite.*overflow"):
            differentiate(f, x, method=method, step=step)
    np.testing.assert_array_equal(x, original)
