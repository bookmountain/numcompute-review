"""Finite-difference differentiation for numerical optimisation.

Provide interfaces for scalar-valued gradients and vector-valued Jacobians
using Python and NumPy. Inputs are one-dimensional real numeric arrays;
derivative outputs are NumPy arrays, and input arrays must be preserved.

Gradients and Jacobians support forward, backward, and central differences.
Method and step-size defaults are provisional API choices.
"""

import numpy as np


def grad(f, x, *, method="central", step=1e-5) -> np.ndarray:
    """Estimate the gradient of a scalar-valued function at x.

    Parameters
    ----------
    f : callable
        Function accepting an array of shape (n,) and returning a finite real
        scalar. It must not mutate its argument.
    x : array-like of shape (n,)
        Nonempty evaluation point containing finite real values. The input
        will not be modified.
    method : {"forward", "backward", "central"}, default="central"
        Finite-difference scheme used for each partial derivative.
    step : float, default=1e-5
        Positive, finite absolute perturbation applied to each coordinate.

    Returns
    -------
    numpy.ndarray of shape (n,)
        Floating-point gradient; entry i approximates df/dx[i].

    Raises
    ------
    TypeError
        If f is not callable, x or f's output is not real numeric data, or
        step is not a real numeric scalar.
    ValueError
        If x is empty, not one-dimensional, or contains non-finite values;
        f returns a non-scalar or non-finite value; method is unsupported;
        step is non-positive or non-finite; or a perturbation is not finite
        or is too small to change a coordinate in floating-point arithmetic.
        Also raised if the computed finite-difference derivative is non-finite,
        possibly due to numerical overflow.

    Notes
    -----
    Forward differences use (f(x + h*e_i) - f(x)) / h; backward differences
    use (f(x) - f(x - h*e_i)) / h; central differences use
    (f(x + h*e_i) - f(x - h*e_i)) / (2*h), where h is step.
    For sufficiently smooth functions, truncation error is O(h) for one-sided
    differences and O(h**2) for central differences. Very small steps can
    amplify floating-point roundoff. Evaluation points use float64 copies.

    Uses 2n function evaluations for central differences and n + 1 for
    forward/backward differences. Copying evaluation points takes O(n**2)
    time in addition to the cost of f. Working/output space is O(n), excluding
    f's allocations. Exceptions raised by f propagate to the caller.
    """
    if not callable(f):
        raise TypeError("f must be callable")
    if not isinstance(method, str) or method not in ("forward", "backward", "central"):
        raise ValueError("method must be 'forward', 'backward', or 'central'")

    h = np.asarray(step)
    if h.ndim != 0 or h.dtype.kind not in "iuf":
        raise TypeError("step must be a real numeric scalar")
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("step must be positive and finite")

    point = np.asarray(x)
    if point.dtype.kind not in "biuf":
        raise TypeError("x must contain real numeric values")
    if point.ndim != 1 or point.size == 0:
        raise ValueError("x must be a nonempty one-dimensional array")
    point = point.astype(float, copy=True)
    if not np.isfinite(point).all():
        raise ValueError("x must contain only finite values")

    def evaluate(values):
        result = np.asarray(f(values))
        if result.dtype.kind not in "biuf":
            raise TypeError("f must return a real numeric scalar")
        if result.ndim != 0:
            raise ValueError("f must return a scalar")
        value = float(result)
        if not np.isfinite(value):
            raise ValueError("f must return a finite value")
        return value

    def perturbed(i, direction):
        values = point.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            values[i] += direction * h
        if not np.isfinite(values[i]):
            raise ValueError("step produces a non-finite evaluation point")
        if values[i] == point[i]:
            raise ValueError("step is too small to change an input coordinate")
        return values

    gradient = np.empty_like(point)
    base = evaluate(point.copy()) if method != "central" else None
    for i in range(point.size):
        if method == "forward":
            gradient[i] = (evaluate(perturbed(i, 1)) - base) / h
        elif method == "backward":
            gradient[i] = (base - evaluate(perturbed(i, -1))) / h
        else:
            upper = evaluate(perturbed(i, 1))
            lower = evaluate(perturbed(i, -1))
            gradient[i] = ((upper - lower) / h) / 2
    if not np.isfinite(gradient).all():
        raise ValueError(
            "finite-difference derivative is non-finite, possibly due to numerical overflow"
        )
    return gradient


def jacobian(f, x, *, method="central", step=1e-5) -> np.ndarray:
    """Estimate the Jacobian of a vector-valued function at x.

    Parameters
    ----------
    f : callable
        Function accepting an array of shape (n,) and returning finite real
        values of shape (m,). The output shape must remain constant and
        nonempty across evaluations. It must not mutate its argument.
    x : array-like of shape (n,)
        Nonempty evaluation point containing finite real values. The input
        will not be modified.
    method : {"forward", "backward", "central"}, default="central"
        Finite-difference scheme used for each partial derivative.
    step : float, default=1e-5
        Positive, finite absolute perturbation applied to each coordinate.

    Returns
    -------
    numpy.ndarray of shape (m, n)
        Floating-point Jacobian; entry (i, j) approximates df[i]/dx[j].
        Rows correspond to outputs and columns to input coordinates.

    Raises
    ------
    TypeError
        If f is not callable, x or f's output is not real numeric data, or
        step is not a real numeric scalar.
    ValueError
        If x is empty, not one-dimensional, or contains non-finite values;
        f returns an empty, non-vector, non-finite, or inconsistent output;
        method is unsupported; step is non-positive or non-finite; or a
        perturbation is not finite or is too small to change a coordinate
        in floating-point arithmetic.
        Also raised if the computed finite-difference derivative is non-finite,
        possibly due to numerical overflow.

    Notes
    -----
    Column j uses (f(x + h*e_j) - f(x)) / h for forward differences,
    (f(x) - f(x - h*e_j)) / h for backward differences, and
    (f(x + h*e_j) - f(x - h*e_j)) / (2*h) for central differences.
    For sufficiently smooth functions, truncation error is O(h) for one-sided
    differences and O(h**2) for central differences. Very small steps can
    amplify floating-point roundoff. Evaluation points and outputs are copied
    to float64 arrays.

    An evaluation at x establishes the output shape. Uses 2n + 1 function
    evaluations for central differences and n + 1 for forward/backward
    differences. Copying points and computing columns takes O(n**2 + m*n)
    time in addition to the cost of f. Working/output space is O(m*n + n + m),
    excluding f's allocations. Exceptions raised by f propagate to the caller.
    """
    if not callable(f):
        raise TypeError("f must be callable")
    if not isinstance(method, str) or method not in ("forward", "backward", "central"):
        raise ValueError("method must be 'forward', 'backward', or 'central'")

    h = np.asarray(step)
    if h.ndim != 0 or h.dtype.kind not in "iuf":
        raise TypeError("step must be a real numeric scalar")
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("step must be positive and finite")

    point = np.asarray(x)
    if point.dtype.kind not in "biuf":
        raise TypeError("x must contain real numeric values")
    if point.ndim != 1 or point.size == 0:
        raise ValueError("x must be a nonempty one-dimensional array")
    point = point.astype(float, copy=True)
    if not np.isfinite(point).all():
        raise ValueError("x must contain only finite values")

    def evaluate(values, expected_shape=None):
        result = np.asarray(f(values))
        if result.dtype.kind not in "biuf":
            raise TypeError("f must return real numeric values")
        if result.ndim != 1 or result.size == 0:
            raise ValueError("f must return a nonempty one-dimensional array")
        if expected_shape is not None and result.shape != expected_shape:
            raise ValueError("f must return the same output shape at every evaluation")
        result = result.astype(float, copy=True)
        if not np.isfinite(result).all():
            raise ValueError("f must return only finite values")
        return result

    def perturbed(j, direction):
        values = point.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            values[j] += direction * h
        if not np.isfinite(values[j]):
            raise ValueError("step produces a non-finite evaluation point")
        if values[j] == point[j]:
            raise ValueError("step is too small to change an input coordinate")
        return values

    base = evaluate(point.copy())
    result = np.empty((base.size, point.size), dtype=float)
    for j in range(point.size):
        if method == "forward":
            upper = evaluate(perturbed(j, 1), base.shape)
            result[:, j] = (upper - base) / h
        elif method == "backward":
            lower = evaluate(perturbed(j, -1), base.shape)
            result[:, j] = (base - lower) / h
        else:
            upper = evaluate(perturbed(j, 1), base.shape)
            lower = evaluate(perturbed(j, -1), base.shape)
            result[:, j] = ((upper - lower) / h) / 2
    if not np.isfinite(result).all():
        raise ValueError(
            "finite-difference derivative is non-finite, possibly due to numerical overflow"
        )
    return result
