"""Tests for phs_metrics() – mirrors test-phs_metrics.R."""

import warnings

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

from pyhazrd import phs_metrics
from pyhazrd.data import load_test_data

TEST_DATA = load_test_data()


# ── Regression anchors ────────────────────────────────────────────────────────


def test_hr_continuous_group_value():
    result = phs_metrics(TEST_DATA, metrics=["HR"], hr_method="continuous_group")
    assert result["estimate"].iloc[0] == pytest.approx(7.33, abs=0.5)


def test_cindex_harrell_value():
    result = phs_metrics(TEST_DATA, metrics=["C_index"], cindex_method="harrell")
    assert 0.5 < result["estimate"].iloc[0] < 1.0


# ── Output structure ──────────────────────────────────────────────────────────


def test_returns_dataframe():
    result = phs_metrics(TEST_DATA)
    assert isinstance(result, pd.DataFrame)


def test_canonical_columns_present():
    canonical = [
        "metric", "estimate", "conf_low", "conf_high", "se",
        "n_numerator", "n_denominator", "method", "adjusted",
    ]
    result = phs_metrics(TEST_DATA)
    assert list(result.columns) == canonical

    result_boot = phs_metrics(TEST_DATA, bootstrap=True, n_boot=50, seed=1)
    assert list(result_boot.columns) == canonical


def test_row_count_single_pair():
    result = phs_metrics(TEST_DATA, metrics=["HR", "C_index"])
    assert len(result) == 2


def test_row_count_multi_hr_pairs():
    pairs = [
        {"numerator": (0.80, 1.00), "denominator": (0.00, 0.20)},
        {"numerator": (0.80, 1.00), "denominator": (0.40, 0.60)},
    ]
    result = phs_metrics(TEST_DATA, metrics=["HR"], hr_pairs=pairs)
    assert len(result) == 2


def test_row_count_or_multiple_ages():
    result = phs_metrics(TEST_DATA, metrics=["OR"], or_age=[60, 70, 80])
    assert len(result) == 3


def test_row_count_or_multi_pairs_and_ages():
    or_pairs = [
        {"numerator": (0.80, 1.00), "denominator": (0.00, 0.20)},
        {"numerator": (0.80, 1.00), "denominator": (0.40, 0.60)},
    ]
    result = phs_metrics(TEST_DATA, metrics=["OR"], or_pairs=or_pairs, or_age=[60, 70])
    assert len(result) == 4


def test_n_numerator_denominator_na_for_cindex():
    result = phs_metrics(TEST_DATA, metrics=["HR", "C_index"])
    ci_row = result[result["metric"] == "C_index"]
    assert pd.isna(ci_row["n_numerator"].iloc[0])
    assert pd.isna(ci_row["n_denominator"].iloc[0])


def test_n_numerator_denominator_populated_for_hr():
    result = phs_metrics(TEST_DATA, metrics=["HR", "C_index"])
    hr_row = result[result["metric"] != "C_index"]
    assert not pd.isna(hr_row["n_numerator"].iloc[0])
    assert not pd.isna(hr_row["n_denominator"].iloc[0])


def test_adjusted_always_false():
    result = phs_metrics(TEST_DATA, metrics=["HR", "C_index"])
    assert result["adjusted"].eq(False).all()


def test_ci_se_na_without_bootstrap():
    result = phs_metrics(TEST_DATA, metrics=["HR", "C_index"])
    assert result["conf_low"].isna().all()
    assert result["conf_high"].isna().all()
    assert result["se"].isna().all()


# ── HR metric naming ──────────────────────────────────────────────────────────


def test_default_hr_metric_name():
    result = phs_metrics(TEST_DATA, metrics=["HR"])
    assert result["metric"].iloc[0] == "HR[80-100]_[0-20]"


def test_custom_hr_numerator_denominator_metric_name():
    result = phs_metrics(TEST_DATA, metrics=["HR"],
                         hr_numerator=0.90, hr_denominator=0.10)
    assert result["metric"].iloc[0] == "HR[90-100]_[0-10]"


def test_hr_pairs_metric_names():
    pairs = [
        {"numerator": (0.80, 1.00), "denominator": (0.00, 0.20)},
        {"numerator": (0.80, 1.00), "denominator": (0.40, 0.60)},
    ]
    result = phs_metrics(TEST_DATA, metrics=["HR"], hr_pairs=pairs)
    assert list(result["metric"]) == ["HR[80-100]_[0-20]", "HR[80-100]_[40-60]"]


def test_hr_pairs_and_numerator_raises():
    with pytest.raises(ValueError, match="not both"):
        phs_metrics(
            TEST_DATA, metrics=["HR"],
            hr_numerator=0.80,
            hr_pairs=[{"numerator": (0.80, 1.00), "denominator": (0.00, 0.20)}],
        )


# ── HR monotonicity ──────────────────────────────────────────────────────────


def test_hr_greater_than_one():
    result = phs_metrics(TEST_DATA, metrics=["HR"])
    assert result["estimate"].iloc[0] > 1


def test_narrow_bands_yield_higher_hr():
    r_wide = phs_metrics(TEST_DATA, metrics=["HR"],
                         hr_numerator=0.80, hr_denominator=0.20)
    r_narrow = phs_metrics(TEST_DATA, metrics=["HR"],
                           hr_numerator=0.90, hr_denominator=0.10)
    assert r_narrow["estimate"].iloc[0] > r_wide["estimate"].iloc[0]


def test_identical_bands_hr_approx_one():
    result = phs_metrics(
        TEST_DATA, metrics=["HR"],
        hr_pairs=[{"numerator": (0.40, 0.60), "denominator": (0.40, 0.60)}],
    )
    assert result["estimate"].iloc[0] == pytest.approx(1.0, abs=0.01)


# ── C-index sanity ────────────────────────────────────────────────────────────


def test_cindex_between_half_and_one():
    result = phs_metrics(TEST_DATA, metrics=["C_index"])
    assert 0.5 < result["estimate"].iloc[0] < 1.0


def test_cindex_method_column():
    result = phs_metrics(TEST_DATA, metrics=["C_index"], cindex_method="harrell")
    assert result["method"].iloc[0] == "harrell"


# ── OR ────────────────────────────────────────────────────────────────────────


def test_or_requires_or_age():
    with pytest.raises(ValueError, match="or_age"):
        phs_metrics(TEST_DATA, metrics=["OR"])


def test_or_metric_name():
    result = phs_metrics(TEST_DATA, metrics=["OR"], or_age=70)
    assert result["metric"].iloc[0] == "OR[80-100]_[0-20]_age70"


def test_or_one_row_per_age():
    result = phs_metrics(TEST_DATA, metrics=["OR"], or_age=[60, 70, 80])
    assert len(result) == 3
    for age in (60, 70, 80):
        assert f"OR[80-100]_[0-20]_age{age}" in list(result["metric"])


def test_or_greater_than_one():
    result = phs_metrics(TEST_DATA, metrics=["OR"], or_age=70)
    assert result["estimate"].iloc[0] > 1


def test_out_of_range_or_age_returns_nan_with_warning():
    with pytest.warns(UserWarning, match="outside"):
        result = phs_metrics(TEST_DATA, metrics=["OR"], or_age=9999)
    assert np.isnan(result["estimate"].iloc[0])


def test_or_pairs_and_numerator_raises():
    with pytest.raises(ValueError, match="not both"):
        phs_metrics(
            TEST_DATA, metrics=["OR"],
            or_age=70,
            or_numerator=0.80,
            or_pairs=[{"numerator": (0.80, 1.00), "denominator": (0.00, 0.20)}],
        )


# ── HR_SD ─────────────────────────────────────────────────────────────────────


def test_hrsd_value():
    result = phs_metrics(TEST_DATA, metrics=["HR_SD"])
    # exp(beta * sd(phs))
    cxph = CoxPHFitter()
    cxph.fit(
        TEST_DATA[["phs", "age", "status"]],
        duration_col="age",
        event_col="status",
    )
    expected = float(np.exp(float(cxph.params_["phs"]) * TEST_DATA["phs"].std()))
    assert result["estimate"].iloc[0] == pytest.approx(expected, rel=0.001)


def test_hrsd_metric_name():
    result = phs_metrics(TEST_DATA, metrics=["HR_SD"])
    assert result["metric"].iloc[0] == "HR_SD"


def test_hrsd_greater_than_one():
    result = phs_metrics(TEST_DATA, metrics=["HR_SD"])
    assert result["estimate"].iloc[0] > 1


def test_hrsd_n_numerator_denominator_na():
    result = phs_metrics(TEST_DATA, metrics=["HR_SD"])
    assert pd.isna(result["n_numerator"].iloc[0])
    assert pd.isna(result["n_denominator"].iloc[0])


# ── Input validation ──────────────────────────────────────────────────────────


def test_missing_column_raises():
    with pytest.raises(ValueError, match="nonexistent"):
        phs_metrics(TEST_DATA, phs="nonexistent")


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="Unknown metric"):
        phs_metrics(TEST_DATA, metrics=["HR", "cindex"])

    with pytest.raises(ValueError, match="Valid options are"):
        phs_metrics(TEST_DATA, metrics=["completely_wrong"])

    with pytest.raises(ValueError, match="cindex"):
        phs_metrics(TEST_DATA, metrics=["cindex", "hr"])
