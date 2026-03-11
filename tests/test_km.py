"""Tests for phs_km_curve() – mirrors test-km.R."""

import numpy as np
import pandas as pd
import pytest

from pyhazrd import phs_km_curve
from pyhazrd.data import load_test_data

TEST_DATA = load_test_data()


# ── phs_km_curve ─────────────────────────────────────────────────────────────


def test_phs_km_curve_returns_figure_by_default():
    import matplotlib.figure

    p = phs_km_curve(TEST_DATA)
    assert isinstance(p, matplotlib.figure.Figure)


def test_phs_km_curve_data_output_has_required_columns():
    result = phs_km_curve(TEST_DATA, output="data")
    assert isinstance(result, pd.DataFrame)
    for col in ("time", "estimate", "conf.low", "conf.high", "stratum"):
        assert col in result.columns


def test_phs_km_curve_default_breaks_three_strata():
    result = phs_km_curve(TEST_DATA, output="data")
    assert len(result["stratum"].unique()) == 3


def test_phs_km_curve_values_clamped_to_0_1():
    result = phs_km_curve(TEST_DATA, output="data")
    assert result["estimate"].between(0, 1).all()
    assert result["conf.low"].between(0, 1).all()
    assert result["conf.high"].between(0, 1).all()


def test_phs_km_curve_two_breaks_three_strata():
    result = phs_km_curve(TEST_DATA, breaks=[0.33, 0.67], output="data")
    assert len(result["stratum"].unique()) == 3


def test_phs_km_curve_three_breaks_four_strata():
    result = phs_km_curve(TEST_DATA, breaks=[0.25, 0.50, 0.75], output="data")
    assert len(result["stratum"].unique()) == 4


def test_phs_km_curve_four_breaks_five_strata():
    result = phs_km_curve(TEST_DATA, breaks=[0.20, 0.40, 0.60, 0.80], output="data")
    assert len(result["stratum"].unique()) == 5


def test_phs_km_curve_conf_int_false_omits_ci():
    result_with = phs_km_curve(TEST_DATA, conf_int=True, output="data")
    result_without = phs_km_curve(TEST_DATA, conf_int=False, output="data")
    # With CI, columns should have real values; without, they should be NaN
    assert result_with["conf.low"].notna().any()
    assert result_without["conf.low"].isna().all()
    assert result_without["conf.high"].isna().all()


def test_phs_km_curve_ref_data_shifts_strata():
    ref = TEST_DATA[TEST_DATA["phs"] <= TEST_DATA["phs"].median()]
    result_ref = phs_km_curve(TEST_DATA, ref_data=ref, output="data")
    result_self = phs_km_curve(TEST_DATA, output="data")
    # Stratum assignments should differ when reference population differs
    merged = result_ref["stratum"].reset_index(drop=True).astype(str)
    merged2 = result_self["stratum"].reset_index(drop=True).astype(str)
    # They may have different lengths; compare what we can
    assert not merged.equals(merged2)


def test_phs_km_curve_invalid_output_raises():
    with pytest.raises(ValueError):
        phs_km_curve(TEST_DATA, output="invalid")


def test_phs_km_curve_breaks_outside_0_1_raises():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        phs_km_curve(TEST_DATA, breaks=[0.0, 0.80])

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        phs_km_curve(TEST_DATA, breaks=[0.20, 1.0])


def test_phs_km_curve_missing_column_raises():
    with pytest.raises(ValueError):
        phs_km_curve(TEST_DATA, phs="nonexistent_col")

