"""Tests for bootstrap functionality – mirrors test-bootstrap.R."""

import numpy as np
import pandas as pd
import pytest

from pyhazrd import phs_metrics
from pyhazrd.data import load_test_data

TEST_DATA = load_test_data()


def test_bootstrap_false_ci_na():
    result = phs_metrics(TEST_DATA, metrics=["HR", "C_index"], bootstrap=False)
    assert result["conf_low"].isna().all()
    assert result["conf_high"].isna().all()
    assert result["se"].isna().all()


def test_bootstrap_fills_ci_and_se():
    result = phs_metrics(TEST_DATA, metrics=["HR", "C_index"],
                         bootstrap=True, n_boot=50, seed=1)
    assert not result["conf_low"].isna().any()
    assert not result["conf_high"].isna().any()
    assert not result["se"].isna().any()


def test_bootstrap_ci_ordered():
    result = phs_metrics(TEST_DATA, metrics=["HR", "C_index"],
                         bootstrap=True, n_boot=50, seed=1)
    assert (result["conf_low"] < result["estimate"]).all()
    assert (result["conf_high"] > result["estimate"]).all()


def test_bootstrap_se_positive():
    result = phs_metrics(TEST_DATA, metrics=["C_index"],
                         bootstrap=True, n_boot=50, seed=1)
    assert (result["se"] > 0).all()


def test_bootstrap_seed_reproducible():
    r1 = phs_metrics(TEST_DATA, metrics=["HR"],
                     bootstrap=True, n_boot=50, seed=42)
    r2 = phs_metrics(TEST_DATA, metrics=["HR"],
                     bootstrap=True, n_boot=50, seed=42)
    assert r1["conf_low"].iloc[0] == pytest.approx(r2["conf_low"].iloc[0])
    assert r1["conf_high"].iloc[0] == pytest.approx(r2["conf_high"].iloc[0])
    assert r1["se"].iloc[0] == pytest.approx(r2["se"].iloc[0])


def test_bootstrap_different_seeds_differ():
    r1 = phs_metrics(TEST_DATA, metrics=["HR"],
                     bootstrap=True, n_boot=50, seed=1)
    r2 = phs_metrics(TEST_DATA, metrics=["HR"],
                     bootstrap=True, n_boot=50, seed=99)
    assert r1["conf_low"].iloc[0] != r2["conf_low"].iloc[0]


def test_bootstrap_point_estimates_unchanged():
    r_plain = phs_metrics(TEST_DATA, metrics=["HR", "C_index"], bootstrap=False)
    r_boot = phs_metrics(TEST_DATA, metrics=["HR", "C_index"],
                         bootstrap=True, n_boot=50, seed=1)
    assert r_plain["estimate"].tolist() == pytest.approx(r_boot["estimate"].tolist())
    assert r_plain["metric"].tolist() == r_boot["metric"].tolist()


def test_bootstrap_multi_hr_pairs():
    pairs = [
        {"numerator": (0.80, 1.00), "denominator": (0.00, 0.20)},
        {"numerator": (0.80, 1.00), "denominator": (0.40, 0.60)},
    ]
    result = phs_metrics(TEST_DATA, metrics=["HR"], hr_pairs=pairs,
                         bootstrap=True, n_boot=50, seed=1)
    assert len(result) == 2
    assert not result["conf_low"].isna().any()
    assert not result["conf_high"].isna().any()


def test_bootstrap_or_multiple_ages():
    result = phs_metrics(TEST_DATA, metrics=["OR"],
                         or_age=[60, 70],
                         bootstrap=True, n_boot=50, seed=1)
    assert len(result) == 2
    assert not result["conf_low"].isna().any()
    assert not result["conf_high"].isna().any()


def test_bootstrap_bca_method():
    result = phs_metrics(TEST_DATA, metrics=["C_index"],
                         bootstrap=True, n_boot=200, seed=7,
                         boot_method="bca")
    assert isinstance(result, pd.DataFrame)
    assert pd.api.types.is_float_dtype(result["conf_low"])
    assert pd.api.types.is_float_dtype(result["conf_high"])
    assert result["se"].notna().all()


def test_bootstrap_normal_method():
    result = phs_metrics(TEST_DATA, metrics=["C_index"],
                         bootstrap=True, n_boot=50, seed=1,
                         boot_method="normal")
    assert not result["conf_low"].isna().any()
    assert not result["conf_high"].isna().any()


def test_bootstrap_ci_level_wider_with_higher_level():
    r95 = phs_metrics(TEST_DATA, metrics=["C_index"],
                      bootstrap=True, n_boot=200, seed=1, ci_level=0.95)
    r50 = phs_metrics(TEST_DATA, metrics=["C_index"],
                      bootstrap=True, n_boot=200, seed=1, ci_level=0.50)
    width95 = float(r95["conf_high"].iloc[0] - r95["conf_low"].iloc[0])
    width50 = float(r50["conf_high"].iloc[0] - r50["conf_low"].iloc[0])
    assert width95 > width50


def test_bootstrap_strata_accepted():
    import numpy as np
    rng = np.random.default_rng(1)
    d = TEST_DATA.copy()
    d["cohort"] = rng.choice(["A", "B"], size=len(d))

    result = phs_metrics(d, metrics=["HR"],
                         bootstrap=True, n_boot=50, seed=1,
                         strata="cohort")
    assert isinstance(result, pd.DataFrame)
    assert not result["conf_low"].isna().any()


def test_bootstrap_parallel_non_no_raises():
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        phs_metrics(TEST_DATA, metrics=["HR"],
                    bootstrap=True, n_boot=10,
                    parallel="multicore")


def test_invalid_boot_method_raises():
    with pytest.raises(ValueError):
        phs_metrics(TEST_DATA, metrics=["HR"],
                    bootstrap=True, n_boot=10,
                    boot_method="jackknife")


def test_bootstrap_hrsd_fills_ci():
    result = phs_metrics(TEST_DATA, metrics=["HR_SD"],
                         bootstrap=True, n_boot=50, seed=1)
    assert not pd.isna(result["conf_low"].iloc[0])
    assert not pd.isna(result["conf_high"].iloc[0])
    assert not pd.isna(result["se"].iloc[0])
    assert result["se"].iloc[0] > 0


def test_bootstrap_out_of_range_or_age_returns_nan():
    with pytest.warns(UserWarning, match="outside"):
        result = phs_metrics(TEST_DATA, metrics=["OR"], or_age=9999,
                             bootstrap=True, n_boot=10, seed=1)
    assert np.isnan(result["estimate"].iloc[0])
