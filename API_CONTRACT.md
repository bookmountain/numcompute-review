# NumCompute API Contract

This document defines the shared interfaces for NumCompute components.

## 1. Transformer interface

A transformer learns parameters from data and transforms input data.

Required methods:

```python
fit(X) -> self
transform(X) -> np.ndarray
fit_transform(X) -> np.ndarray
```

Rules:

- `fit(X)` learns parameters from `X` and returns `self`.
- `transform(X)` uses the parameters learned during `fit`.
- `transform(X)` must not re-fit or change learned parameters.
- `fit_transform(X)` is equivalent to `fit(X).transform(X)`.
- Outputs must be NumPy arrays.
- Input and output shapes must be documented.
- Calling `transform(X)` before `fit(X)` should raise a clear error.

## 2. Estimator interface

An estimator learns from input data and produces predictions.

Required methods:

```python
fit(X, y=None) -> self
predict(X) -> np.ndarray
```

Rules:

- `fit(X, y)` learns the estimator parameters and returns `self`.
- `predict(X)` uses the fitted estimator.
- Calling `predict(X)` before `fit(X)` should raise a clear error.
- Predictions must be returned as NumPy arrays.
- The estimator must validate the number of input features.

## 3. Pipeline interface

A pipeline contains ordered processing steps.

Example:

```python
pipe = Pipeline([
    ("imputer", imputer),
    ("scaler", scaler),
])
```

Required behaviour:

- A pipeline requires at least one named step. Step names must be unique.
- Every non-final step must provide callable `fit(X)` and `transform(X)` methods.
- The final step may be a transformer or an estimator, following the interfaces
  above.
- Input `X` must be a two-dimensional NumPy array with shape
  `(n_samples, n_features)`.
- `fit(X, y=None)` fits each step in order, passing each non-final step's
  transformed output to the next step. Transformers receive `X`; a final
  estimator receives the transformed `X` and `y`.
- `transform(X)` is valid only when the final step provides `transform()`.
  It applies each transformer in order and returns the transformed data.
- `predict(X)` is valid only when the final step provides `predict()`.
  It transforms `X` through the non-final steps, then returns the final step's
  predictions.
- `fit_transform(X, y=None)` fits and transforms the pipeline and is valid only
  when the final step provides `transform()`.
- Transformations must return two-dimensional NumPy arrays and preserve the
  number of rows (`n_samples`), but may change the number of features.
- Calling `transform()` or `predict()` before `fit()` raises a clear fitted-state
  error. These methods use fitted state without re-fitting any step.
- Calling an operation unsupported by the final step raises a clear error.
- Supplied step objects are fitted directly and are not copied.
- Invalid steps must raise a clear error.

## 4. FeatureUnion interface

A feature union combines the outputs of named transformer branches and follows
the transformer interface: `fit(X) -> self`, `transform(X) -> np.ndarray`, and
`fit_transform(X) -> np.ndarray`.

Required behaviour:

- A feature union requires at least one named transformer branch. Branch names
  must be unique, and each branch must follow the transformer interface above.
- Input `X` must be a two-dimensional NumPy array with shape
  `(n_samples, n_features)`.
- Every branch receives the same `X`, rather than the output of another branch.
- `fit(X)` fits each supplied branch object directly without copying it and
  returns `self`.
- `transform(X)` uses each branch's fitted state without re-fitting. Each branch
  must return a two-dimensional NumPy array with `n_samples` rows; the number of
  output features may differ between branches.
- Outputs are concatenated column-wise in branch order, producing shape
  `(n_samples, sum_of_branch_output_features)`.
- `fit_transform(X)` is equivalent to `fit(X).transform(X)`.
- Calling `transform(X)` before `fit(X)` raises a clear fitted-state error.
- Invalid branches or incompatible output shapes raise clear errors.

The minimal pipeline and feature-union API does not include callable adapters,
column selectors, or other optional composition features at this stage.

## 5. General conventions

- Use NumPy arrays for numerical data.
- Do not use pandas, scikit-learn, PyTorch, or other external ML libraries.
- Preserve input arrays unless mutation is explicitly documented.
- Document parameter types, return types, shapes, exceptions, and complexity.
- Use clear and consistent error messages.

## 6. Current metrics behavior and open API decisions

These are observed behaviors of the integrated implementation; they are not
requirements specified elsewhere in this contract:

- `roc_curve` preserves exact integer score thresholds by returning an
  object-dtype threshold array when integer scores are used. NumPy's
  `np.isfinite` does not support that array dtype and raises `TypeError`.
  Whether to keep exact integer thresholds or use a numeric dtype with a
  representability trade-off remains an API design decision.
- Weighted `f1` with `zero_division=np.nan` currently returns `NaN` if
  an included class has zero support. The class's undefined score propagates
  through the weighted calculation despite its zero support. Whether to omit
  zero-support classes from that average remains an API design decision.

