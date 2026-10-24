# %% [markdown]
# # NumCompute Quickstart
#
# From the repository root, run the script and notebook checker:
#   python3 demo/quickstart.py
#   python3 demo/check_notebook.py
#
# To run the notebook interactively in VS Code:
# 1. Open this repository folder in VS Code.
# 2. Open demo/quickstart.ipynb.
# 3. Select the existing Python (AIML Course) kernel.
# 4. Choose Run All to execute every cell in order.
#
# This end-to-end tour uses only the APIs implemented in this repository and
# the shared sample CSV. Timings are specific to your machine and input size.

# %% [markdown]
# ## 1. Load the shared CSV
#
# The sample has six training rows and two test rows. Missing numeric values
# are retained as NaN so we can inspect and impute them explicitly.

# %%
from pathlib import Path
import sys
import timeit

import numpy as np

ROOT = next(path for path in (Path.cwd(), *Path.cwd().parents)
            if (path / "numcompute").is_dir())
sys.path.insert(0, str(ROOT))

from numcompute.io import load_csv
from numcompute.metrics import accuracy, confusion_matrix, f1
from numcompute.optim import grad, jacobian
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from numcompute.rank import percentile, rank
from numcompute.sort_search import binary_search, quickselect, stable_sort, topk
from numcompute.stats import histogram, mean, quantile, var

sample_path = ROOT / "demo" / "data" / "sample.csv"
data = load_csv(sample_path, skip_header=1, missing="propagate")
missing_per_column = np.isnan(data).sum(axis=0)
print("CSV shape:", data.shape)
print("Columns: price, weight, category_code")
print("Missing values per column:", missing_per_column.tolist())

# %% [markdown]
# ## 2. Impute, scale, and encode
#
# Fit transformations on training rows only, then reuse the learned values on
# test rows. The pipeline applies the numeric imputer before the scaler.
# Category 2 is absent from training and is ignored by the fitted encoder.

# %%
train, test = data[:6], data[6:]
numeric_train, numeric_test = train[:, :2], test[:, :2]
category_train, category_test = train[:, 2:3], test[:, 2:3]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
])
try:
    numeric_pipeline.transform(numeric_train)
except RuntimeError as error:
    print("Unfitted pipeline check:", error)
else:
    raise AssertionError("Pipeline.transform should require fit first")

scaled_train = numeric_pipeline.fit_transform(numeric_train)
scaled_test = numeric_pipeline.transform(numeric_test)
encoder = OneHotEncoder(handle_unknown="ignore")
encoded_train = encoder.fit_transform(category_train)
encoded_test = encoder.transform(category_test)
prepared_train = np.concatenate((scaled_train, encoded_train), axis=1)
prepared_test = np.concatenate((scaled_test, encoded_test), axis=1)
assert prepared_train.shape == (6, 4)
assert prepared_test.shape == (2, 4)
assert np.isfinite(prepared_train).all() and np.isfinite(prepared_test).all()
print("Scaled numeric train/test:", scaled_train.shape, scaled_test.shape)
print("Encoded categories train/test:", encoded_train.shape, encoded_test.shape)
print("Combined train/test:", prepared_train.shape, prepared_test.shape)
print("Imputer training means:", numeric_pipeline.steps[0][1].statistics_.tolist())
print("Learned categories:", encoder.categories_[0].tolist())
assert np.array_equal(encoded_test[0], [0.0, 0.0])
print("Unknown test category maps to zeros:", encoded_test[0].tolist())

# %% [markdown]
# ## 3. Sort, select, search, and rank
#
# Reuse the imputer fitted above to fill prices, then demonstrate stable
# sorting, deterministic top-k indexes, quickselect, leftmost binary search,
# tie-aware ranking, and percentiles.

# %%
filled_numeric = numeric_pipeline.steps[0][1].transform(data[:, :2])
prices = filled_numeric[:, 0]
sorted_prices = stable_sort(prices)
top_values, top_indices = topk(prices, 3)
third_smallest = quickselect(prices, 2)
search_index, found = binary_search(sorted_prices, 20.0)
price_ranks = rank(prices, method="average")
price_percentiles = percentile(prices, [25, 50, 75])
print("Sorted prices:", sorted_prices.tolist())
print("Three largest (value, original row):", list(zip(top_values.tolist(), top_indices.tolist())))
print("Third-smallest price:", float(third_smallest))
print("Search for 20.0 (left insertion index, found):", (search_index, found))
print("Average ranks, with ties sharing rank:", price_ranks.tolist())
print("25th/50th/75th percentiles:", price_percentiles.tolist())
assert found and sorted_prices[search_index] == 20.0
assert np.count_nonzero(prices == 20.0) == 2

# %% [markdown]
# ## 4. Summarise values and evaluate simple predictions
#
# Statistics use the imputed price column. The small example predictions
# illustrate accuracy, a confusion matrix, and macro F1 without involving
# undefined zero-support classes or optional NaN averaging policies.

# %%
print("Price mean and sample variance:", float(mean(prices)), float(var(prices, ddof=1)))
print("Price quartiles:", quantile(prices, [0.25, 0.5, 0.75]).tolist())
counts, edges = histogram(prices, bins=[0, 20, 30, 50])
print("Histogram counts and edges:", counts.tolist(), edges.tolist())

true_categories = data[:, 2].astype(int)
predicted_categories = np.array([0, 1, 1, 1, 0, 1, 0, 0])
print("Category accuracy:", float(accuracy(true_categories, predicted_categories)))
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(true_categories, predicted_categories, labels=[0, 1, 2]))
print("Macro F1:", float(f1(true_categories, predicted_categories,
                              labels=[0, 1, 2], average="macro")))

# %% [markdown]
# ## 5. Estimate derivatives with known answers
#
# Central finite differences approximate the gradient of a quadratic and the
# Jacobian of a two-output function. Their exact derivatives are shown below.

# %%
point = np.array([3.0, 4.0])
gradient = grad(lambda x: x[0] ** 2 + x[1] ** 2, point)
expected_gradient = np.array([6.0, 8.0])
np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-8, atol=1e-8)
print("Gradient:", gradient.tolist(), " exact:", expected_gradient.tolist())

jacobian_point = np.array([2.0, 3.0])
def vector_function(x):
    return np.array([x[0] ** 2 + x[1], x[0] * x[1]])
jacobian_value = jacobian(vector_function, jacobian_point)
expected_jacobian = np.array([[4.0, 1.0], [3.0, 2.0]])
np.testing.assert_allclose(jacobian_value, expected_jacobian, rtol=1e-8, atol=1e-8)
print("Jacobian:\n", jacobian_value)
print("Exact Jacobian:\n", expected_jacobian)

# %% [markdown]
# ## 6. Compare a Python loop with vectorised NumPy
#
# A fixed seed makes the input reproducible. We check numerical agreement
# before measuring. The table reports this run's median time per call; timings
# vary by machine, Python/NumPy version, and load, so neither method is always
# faster for every workload.

# %%
rng = np.random.default_rng(2026)
values = rng.normal(size=2500)

def loop_sum_squares(array):
    return sum(float(value) * float(value) for value in array)

def numpy_sum_squares(array):
    return np.sum(array * array)

loop_result = loop_sum_squares(values)
numpy_result = numpy_sum_squares(values)
np.testing.assert_allclose(loop_result, numpy_result, rtol=1e-12, atol=1e-12)
loop_times = timeit.repeat(lambda: loop_sum_squares(values), repeat=5, number=3)
numpy_times = timeit.repeat(lambda: numpy_sum_squares(values), repeat=5, number=3)
loop_median = float(np.median(loop_times)) / 3
numpy_median = float(np.median(numpy_times)) / 3
print("Seed: 2026; values: 2500; both methods agree:",
      bool(np.isclose(loop_result, numpy_result, rtol=1e-12, atol=1e-12)))
print("Method                 median seconds/call")
print(f"Python loop            {loop_median:.8f}")
print(f"Vectorised NumPy       {numpy_median:.8f}")
print("Timing is one environment-specific observation, not a universal ranking.")
