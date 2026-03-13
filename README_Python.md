pyhazrd
================

<!-- badges: start -->

[![pytest](https://github.com/espenhgn/hazrd/actions/workflows/pytest.yaml/badge.svg)](https://github.com/espenhgn/hazrd/actions/workflows/pytest.yaml)

<!-- badges: end -->

`pyhazrd` is a Python package for evaluating and visualizing Polygenic Hazard
Score (PHS) analyses. It provides a consistent API for computing
survival-based discrimination metrics, bootstrapped confidence intervals,
Kaplan-Meier curves, and Cox model survival curves.

## Installation

```bash
pip install pyhazrd
```

Or install directly from the repository:

```bash
pip install git+https://github.com/espenhgn/hazrd.git
```

## Requirements

Python ≥ 3.11. Core dependencies (`lifelines`, `scikit-survival`, `numpy`,
`pandas`, `scipy`, `matplotlib`) are installed automatically.

## Quick example

```python
from pyhazrd import phs_metrics, phs_km_curve, phs_cox_curve
from pyhazrd.data import load_test_data

data = load_test_data()

# Compute HR, C-index, OR, and HR_SD in one call
phs_metrics(
    data,
    metrics=["HR", "C_index", "OR", "HR_SD"],
    or_age=70,
)

# Bootstrapped confidence intervals
phs_metrics(
    data,
    metrics=["HR", "C_index"],
    bootstrap=True,
    n_boot=999,
    seed=42,
)

# Kaplan-Meier plot stratified by PHS percentile (returns a matplotlib Figure)
fig = phs_km_curve(data)

# Return tidy data for a custom plot
km_data = phs_km_curve(data, breaks=[0.20, 0.40, 0.60, 0.80], output="data")

# Cox model survival curves at specified PHS percentiles
fig = phs_cox_curve(data, percentiles=[0.01, 0.10, 0.50, 0.90, 0.99])

# Return tidy data from Cox curves
cox_data = phs_cox_curve(data, output="data")
```

## Getting started

A full walkthrough is available in the [Jupyter notebook](jupyter/hazrd.ipynb),
which mirrors the R package vignette and demonstrates all primary functions.
