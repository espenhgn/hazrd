"""Tests for deprecated wrapper functions – mirrors test-deprecated.R."""

import warnings

import pytest

from pyhazrd.data import load_test_data

TEST_DATA = load_test_data()


def test_get_hr_deprecation_warning():
    from pyhazrd.deprecated import get_hr

    with pytest.warns(DeprecationWarning):
        get_hr(TEST_DATA)


def test_get_hr_returns_dict_with_expected_keys():
    from pyhazrd.deprecated import get_hr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = get_hr(TEST_DATA)

    assert isinstance(result, dict)
    assert "index" in result
    assert "value" in result
    assert "conf.low" in result
    assert "conf.high" in result
    assert "iters" in result


def test_get_hr_value_matches_phs_metrics():
    from pyhazrd.deprecated import get_hr
    from pyhazrd import phs_metrics

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = get_hr(TEST_DATA)

    new = phs_metrics(TEST_DATA, metrics=["HR"])
    assert old["value"] == pytest.approx(float(new["estimate"].iloc[0]), rel=1e-6)


def test_get_cindex_deprecation_warning():
    from pyhazrd.deprecated import get_cindex

    with pytest.warns(DeprecationWarning):
        get_cindex(TEST_DATA)


def test_get_cindex_returns_dict():
    from pyhazrd.deprecated import get_cindex

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = get_cindex(TEST_DATA)

    assert isinstance(result, dict)
    assert "value" in result


def test_get_hrsd_deprecation_warning():
    from pyhazrd.deprecated import get_hrsd

    with pytest.warns(DeprecationWarning):
        get_hrsd(TEST_DATA)


def test_get_hrsd_returns_dict():
    from pyhazrd.deprecated import get_hrsd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = get_hrsd(TEST_DATA)

    assert isinstance(result, dict)
    assert "value" in result


def test_get_or_deprecation_warning():
    from pyhazrd.deprecated import get_or

    with pytest.warns(DeprecationWarning):
        get_or(TEST_DATA, or_age=70)


def test_get_or_requires_or_age():
    from pyhazrd.deprecated import get_or

    with pytest.raises((ValueError, TypeError)):
        get_or(TEST_DATA)


def test_get_or_returns_dict():
    from pyhazrd.deprecated import get_or

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = get_or(TEST_DATA, or_age=70)

    assert isinstance(result, dict)
    assert "value" in result
