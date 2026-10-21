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

- Intermediate steps must implement `fit` and `transform`.
- `fit(X, y=None)` fits each step in order.
- `transform(X)` applies each transformer in order.
- `fit_transform(X, y=None)` fits and transforms the pipeline.
- If the final step is a transformer, the pipeline returns transformed data.
- If the final step is an estimator, the pipeline supports `predict(X)`.
- Step names must be unique.
- Invalid steps must raise a clear error.

## 4. General conventions

- Use NumPy arrays for numerical data.
- Do not use pandas, scikit-learn, PyTorch, or other external ML libraries.
- Preserve input arrays unless mutation is explicitly documented.
- Document parameter types, return types, shapes, exceptions, and complexity.
- Use clear and consistent error messages.
```

