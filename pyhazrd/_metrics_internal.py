"""Internal metric calculation helpers for pyhazrd.

Each ``_calc_*`` function computes a single metric and returns a one-row
``pandas.DataFrame`` with the canonical columns:

    metric, estimate, conf_low, conf_high, se,
    n_numerator, n_denominator, method, adjusted
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# _calc_hr_metric
# ---------------------------------------------------------------------------


def _calc_hr_metric(
    data: pd.DataFrame,
    phs: str,
    time: str,
    event: str,
    hr_method: str,
    numerator: tuple[float, float],
    denominator: tuple[float, float],
    cxph: Any,
) -> pd.DataFrame:
    """Compute one hazard-ratio row.

    Parameters
    ----------
    data:
        Full data-frame.
    phs, time, event:
        Column name strings.
    hr_method:
        Only ``"continuous_group"`` is implemented.
    numerator, denominator:
        Length-2 tuples of (lower_percentile, upper_percentile) in [0, 1].
    cxph:
        Fitted :class:`lifelines.CoxPHFitter` instance.
    """
    phs_vals = data[phs].to_numpy()

    if hr_method == "continuous_group":
        cut_num_lo = np.quantile(phs_vals, numerator[0])
        cut_num_hi = np.quantile(phs_vals, numerator[1])
        cut_den_lo = np.quantile(phs_vals, denominator[0])
        cut_den_hi = np.quantile(phs_vals, denominator[1])

        beta = float(cxph.params_[phs])

        ix_num = np.where((phs_vals >= cut_num_lo) & (phs_vals <= cut_num_hi))[0]
        ix_den = np.where((phs_vals >= cut_den_lo) & (phs_vals <= cut_den_hi))[0]

        beta_phs = beta * phs_vals
        estimate = float(
            np.exp(beta_phs[ix_num].mean() - beta_phs[ix_den].mean())
        )

        metric_name = (
            f"HR[{round(numerator[0] * 100)}-{round(numerator[1] * 100)}]"
            f"_[{round(denominator[0] * 100)}-{round(denominator[1] * 100)}]"
        )

    elif hr_method == "continuous_point":
        beta = float(cxph.params_[phs])
        phs_num = float(np.quantile(phs_vals, numerator[0]))
        phs_den = float(np.quantile(phs_vals, denominator[1]))
        estimate = float(np.exp(beta * (phs_num - phs_den)))

        ix_num = np.where(phs_vals >= np.quantile(phs_vals, numerator[0]))[0]
        ix_den = np.where(phs_vals <= np.quantile(phs_vals, denominator[1]))[0]

        metric_name = (
            f"HR{round(numerator[0] * 100)}_{round(denominator[1] * 100)}"
        )

    else:
        raise NotImplementedError(f"hr_method '{hr_method}' not yet implemented")

    return pd.DataFrame(
        {
            "metric": [metric_name],
            "estimate": [estimate],
            "conf_low": [np.nan],
            "conf_high": [np.nan],
            "se": [np.nan],
            "n_numerator": [len(ix_num)],
            "n_denominator": [len(ix_den)],
            "method": [hr_method],
            "adjusted": [False],
        }
    )


# ---------------------------------------------------------------------------
# _calc_cindex_metric
# ---------------------------------------------------------------------------


def _calc_cindex_metric(
    data: pd.DataFrame,
    phs: str,
    time: str,
    event: str,
    cindex_method: str,
    cxph: Any,
) -> pd.DataFrame:
    """Return a one-row DataFrame for the C-index metric."""
    if cindex_method == "harrell":
        estimate = float(cxph.concordance_index_)
    else:
        raise NotImplementedError(f"cindex_method '{cindex_method}' not yet implemented")

    return pd.DataFrame(
        {
            "metric": ["C_index"],
            "estimate": [estimate],
            "conf_low": [np.nan],
            "conf_high": [np.nan],
            "se": [np.nan],
            "n_numerator": [pd.NA],
            "n_denominator": [pd.NA],
            "method": [cindex_method],
            "adjusted": [False],
        }
    )


# ---------------------------------------------------------------------------
# _calc_hrsd_metric
# ---------------------------------------------------------------------------


def _calc_hrsd_metric(
    data: pd.DataFrame,
    phs: str,
    time: str,
    event: str,
    cxph: Any,
) -> pd.DataFrame:
    """Return a one-row DataFrame for the HR-per-SD metric."""
    beta = float(cxph.params_[phs])
    phs_sd = float(data[phs].std(ddof=1))
    estimate = float(np.exp(beta * phs_sd))

    return pd.DataFrame(
        {
            "metric": ["HR_SD"],
            "estimate": [estimate],
            "conf_low": [np.nan],
            "conf_high": [np.nan],
            "se": [np.nan],
            "n_numerator": [pd.NA],
            "n_denominator": [pd.NA],
            "method": [pd.NA],
            "adjusted": [False],
        }
    )


# ---------------------------------------------------------------------------
# _calc_or_metric
# ---------------------------------------------------------------------------


def _calc_or_metric(
    data: pd.DataFrame,
    phs: str,
    time: str,
    event: str,
    numerator: tuple[float, float],
    denominator: tuple[float, float],
    or_age: float,
) -> pd.DataFrame:
    """Return a one-row DataFrame for the OR metric at a given age."""
    from lifelines import KaplanMeierFitter

    phs_vals = data[phs].to_numpy()

    cut_num_lo = np.quantile(phs_vals, numerator[0])
    cut_num_hi = np.quantile(phs_vals, numerator[1])
    cut_den_lo = np.quantile(phs_vals, denominator[0])
    cut_den_hi = np.quantile(phs_vals, denominator[1])

    ix_num = np.where((phs_vals >= cut_num_lo) & (phs_vals <= cut_num_hi))[0]
    ix_den = np.where((phs_vals >= cut_den_lo) & (phs_vals <= cut_den_hi))[0]

    data_num = data.iloc[ix_num]
    data_den = data.iloc[ix_den]

    def _km_surv_at(df: pd.DataFrame, age_eval: float):
        """Return KM survival probability at age_eval; NaN when out of range."""
        kmf = KaplanMeierFitter()
        kmf.fit(df[time], event_observed=df[event])
        km_times = kmf.survival_function_.index.to_numpy()
        km_surv = kmf.survival_function_["KM_estimate"].to_numpy()

        if len(km_times) == 0:
            return float("nan")
        if age_eval < km_times.min() or age_eval > km_times.max():
            return float("nan")

        # Step-function interpolation (same as R approx with rule=1)
        return float(np.interp(age_eval, km_times, km_surv))

    p_num = _km_surv_at(data_num, or_age)
    p_den = _km_surv_at(data_den, or_age)

    metric_name = (
        f"OR[{round(numerator[0] * 100)}-{round(numerator[1] * 100)}]"
        f"_[{round(denominator[0] * 100)}-{round(denominator[1] * 100)}]"
        f"_age{or_age}"
    )

    if np.isnan(p_num) or np.isnan(p_den):
        warnings.warn(
            f"or_age {or_age} is outside the observed time range for one or "
            "both groups. OR cannot be computed.",
            stacklevel=4,
        )
        estimate = float("nan")
    elif p_num <= 0 or p_den <= 0:
        warnings.warn(
            f"or_age {or_age} results in zero survival probability for one or "
            "both groups; OR undefined.",
            stacklevel=4,
        )
        estimate = float("nan")
    else:
        odds_num = (1 - p_num) / p_num
        odds_den = (1 - p_den) / p_den
        estimate = float(odds_num / odds_den)

    return pd.DataFrame(
        {
            "metric": [metric_name],
            "estimate": [estimate],
            "conf_low": [np.nan],
            "conf_high": [np.nan],
            "se": [np.nan],
            "n_numerator": [len(ix_num)],
            "n_denominator": [len(ix_den)],
            "method": [pd.NA],
            "adjusted": [False],
        }
    )
