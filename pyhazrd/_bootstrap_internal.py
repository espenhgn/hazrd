"""Bootstrap orchestration for pyhazrd.

``_boot_statistic``  – evaluated on each resample; returns a dict of metric
                       name → float estimate.
``_run_bootstrap``   – drives the resampling loop and derives CI / SE from the
                       distribution of replicate estimates.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from pyhazrd._metrics_internal import (
    _calc_cindex_metric,
    _calc_hr_metric,
    _calc_hrsd_metric,
    _calc_or_metric,
)


# ---------------------------------------------------------------------------
# _boot_statistic
# ---------------------------------------------------------------------------


def _boot_statistic(
    data: pd.DataFrame,
    indices: np.ndarray,
    phs: str,
    time: str,
    event: str,
    metrics: list[str],
    hr_method: str,
    hr_pairs: list[dict],
    cindex_method: str,
    or_age: list[float],
    or_pairs: list[dict],
) -> dict[str, float]:
    """Compute all requested metrics on one bootstrap resample.

    Parameters
    ----------
    data:
        Full data-frame.
    indices:
        Row indices for this resample.
    phs, time, event:
        Column name strings.
    metrics:
        Which metrics to compute.
    hr_method, hr_pairs, cindex_method, or_age, or_pairs:
        Forwarded from :func:`phs_metrics`.

    Returns
    -------
    dict mapping metric name → float estimate (NaN on failure).
    """
    from lifelines import CoxPHFitter

    d = data.iloc[indices].reset_index(drop=True).copy()
    estimates: dict[str, float] = {}

    needs_coxph = any(m in metrics for m in ("HR", "C_index", "HR_SD"))
    cxph = None
    if needs_coxph:
        try:
            cxph = CoxPHFitter()
            cxph.fit(d[[phs, time, event]], duration_col=time, event_col=event)
        except Exception:
            cxph = None

    if "HR" in metrics:
        for pair in hr_pairs:
            try:
                row = _calc_hr_metric(
                    d, phs, time, event,
                    hr_method, pair["numerator"], pair["denominator"], cxph
                )
                estimates[row["metric"].iloc[0]] = float(row["estimate"].iloc[0])
            except Exception:
                # Build the expected metric name so vector length stays stable
                num = pair["numerator"]
                den = pair["denominator"]
                nm = (
                    f"HR[{round(num[0]*100)}-{round(num[1]*100)}]"
                    f"_[{round(den[0]*100)}-{round(den[1]*100)}]"
                )
                estimates[nm] = float("nan")

    if "C_index" in metrics:
        try:
            row = _calc_cindex_metric(d, phs, time, event, cindex_method, cxph)
            estimates["C_index"] = float(row["estimate"].iloc[0])
        except Exception:
            estimates["C_index"] = float("nan")

    if "HR_SD" in metrics:
        try:
            row = _calc_hrsd_metric(d, phs, time, event, cxph)
            estimates["HR_SD"] = float(row["estimate"].iloc[0])
        except Exception:
            estimates["HR_SD"] = float("nan")

    if "OR" in metrics:
        for pair in or_pairs:
            for age in or_age:
                num = pair["numerator"]
                den = pair["denominator"]
                nm = (
                    f"OR[{round(num[0]*100)}-{round(num[1]*100)}]"
                    f"_[{round(den[0]*100)}-{round(den[1]*100)}]"
                    f"_age{age}"
                )
                try:
                    row = _calc_or_metric(
                        d, phs, time, event,
                        pair["numerator"], pair["denominator"], age
                    )
                    estimates[nm] = float(row["estimate"].iloc[0])
                except Exception:
                    estimates[nm] = float("nan")

    return estimates


# ---------------------------------------------------------------------------
# _run_bootstrap
# ---------------------------------------------------------------------------


def _run_bootstrap(
    point_estimates: pd.DataFrame,
    data: pd.DataFrame,
    phs: str,
    time: str,
    event: str,
    metrics: list[str],
    hr_method: str,
    hr_pairs: list[dict],
    cindex_method: str,
    or_age: list[float],
    or_pairs: list[dict],
    n_boot: int = 999,
    ci_level: float = 0.95,
    boot_method: str = "percentile",
    seed: int | None = None,
    parallel: str = "no",
    n_cores: int = 1,
    strata_col: str | None = None,
) -> pd.DataFrame:
    """Run bootstrap resampling and merge CI / SE into *point_estimates*.

    Parameters
    ----------
    point_estimates:
        Tibble returned by the non-bootstrap path of :func:`phs_metrics`.
    data, phs, time, event, metrics, hr_method, hr_pairs,
    cindex_method, or_age, or_pairs:
        Forwarded to :func:`_boot_statistic`.
    n_boot:
        Number of replicates.
    ci_level:
        Confidence level (e.g. ``0.95``).
    boot_method:
        One of ``"percentile"``, ``"bca"``, or ``"normal"``.
    seed:
        Optional integer for reproducibility.
    parallel:
        Only ``"no"`` is implemented.
    n_cores:
        Ignored (reserved).
    strata_col:
        Optional column name; resampling is stratified within each level.

    Returns
    -------
    The same ``DataFrame`` as *point_estimates* with ``conf_low``,
    ``conf_high``, and ``se`` columns filled.
    """
    if parallel != "no":
        raise NotImplementedError(
            f"parallel='{parallel}' is not yet implemented. "
            "Use parallel='no' for now."
        )

    valid_boot_methods = ("percentile", "bca", "normal")
    if boot_method not in valid_boot_methods:
        raise ValueError(
            f"boot_method must be one of {valid_boot_methods}, got '{boot_method}'"
        )

    rng = np.random.default_rng(seed)

    n = len(data)

    # Build strata index groups for stratified resampling
    if strata_col is not None:
        if strata_col not in data.columns:
            raise ValueError(f"strata column '{strata_col}' not found in data.")
        strata_vals = data[strata_col].to_numpy()
        strata_levels = np.unique(strata_vals)
        stratum_indices = [np.where(strata_vals == lvl)[0] for lvl in strata_levels]
    else:
        stratum_indices = [np.arange(n)]

    # Collect replicate estimates: list of dicts, each dict is metric→float
    boot_results: list[dict[str, float]] = []

    for _ in range(n_boot):
        # Stratified resampling: within each stratum resample with replacement
        idx_parts = [
            rng.choice(grp, size=len(grp), replace=True)
            for grp in stratum_indices
        ]
        indices = np.concatenate(idx_parts)

        result = _boot_statistic(
            data=data,
            indices=indices,
            phs=phs,
            time=time,
            event=event,
            metrics=metrics,
            hr_method=hr_method,
            hr_pairs=hr_pairs,
            cindex_method=cindex_method,
            or_age=or_age,
            or_pairs=or_pairs,
        )
        boot_results.append(result)

    # Convert to dict of metric → ndarray of replicates
    all_metric_names = list(point_estimates["metric"])
    boot_arrays: dict[str, np.ndarray] = {
        nm: np.array([r.get(nm, float("nan")) for r in boot_results], dtype=float)
        for nm in all_metric_names
    }

    # Compute CI and SE for each metric
    alpha = 1.0 - ci_level
    ci_rows = []

    for nm in all_metric_names:
        arr = boot_arrays[nm]
        valid = arr[~np.isnan(arr)]
        t0 = float(
            point_estimates.loc[point_estimates["metric"] == nm, "estimate"].iloc[0]
        )

        se_val = float(np.std(arr, ddof=1)) if len(valid) > 1 else float("nan")

        if len(valid) < 2:
            ci_rows.append(
                {"metric": nm, "conf_low": float("nan"), "conf_high": float("nan"), "se": se_val}
            )
            continue

        if boot_method == "percentile":
            lo = float(np.nanpercentile(arr, 100 * alpha / 2))
            hi = float(np.nanpercentile(arr, 100 * (1 - alpha / 2)))

        elif boot_method == "normal":
            z = float(_norm.ppf(1 - alpha / 2))
            lo = t0 - z * se_val
            hi = t0 + z * se_val

        elif boot_method == "bca":
            lo, hi = _bca_ci(arr, t0, alpha)

        ci_rows.append({"metric": nm, "conf_low": lo, "conf_high": hi, "se": se_val})

    ci_df = pd.DataFrame(ci_rows)

    # Merge back into point_estimates
    result_df = point_estimates.copy()
    merged = result_df.merge(ci_df, on="metric", how="left", suffixes=(".orig", ""))

    for col in ("conf_low", "conf_high", "se"):
        orig = col + ".orig"
        if orig in merged.columns:
            merged.drop(columns=[orig], inplace=True)

    canonical = [
        "metric", "estimate", "conf_low", "conf_high", "se",
        "n_numerator", "n_denominator", "method", "adjusted",
    ]
    return merged[canonical].reset_index(drop=True)


# ---------------------------------------------------------------------------
# BCA helper
# ---------------------------------------------------------------------------


def _bca_ci(
    boot_dist: np.ndarray,
    t0: float,
    alpha: float,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) confidence interval.

    Uses a simplified jackknife approach for the acceleration constant.
    """
    valid = boot_dist[~np.isnan(boot_dist)]
    if len(valid) < 2:
        return float("nan"), float("nan")

    # Bias-correction factor z0
    prop_below = np.mean(valid < t0)
    if prop_below <= 0 or prop_below >= 1:
        # Fall back to percentile when all replicates are on one side
        lo = float(np.nanpercentile(boot_dist, 100 * alpha / 2))
        hi = float(np.nanpercentile(boot_dist, 100 * (1 - alpha / 2)))
        return lo, hi

    z0 = float(_norm.ppf(prop_below))

    # Acceleration via leave-one-out mean differences (simplified jackknife on boot dist)
    jk_mean = np.mean(valid)
    jk_diffs = jk_mean - valid
    num = np.sum(jk_diffs**3)
    den = 6.0 * (np.sum(jk_diffs**2) ** 1.5)
    a = float(num / den) if abs(den) > 1e-10 else 0.0

    z_lo = _norm.ppf(alpha / 2)
    z_hi = _norm.ppf(1 - alpha / 2)

    def _adj(z: float) -> float:
        denom = 1.0 - a * (z0 + z)
        if abs(denom) < 1e-10:
            return z
        return float(_norm.cdf(z0 + (z0 + z) / denom))

    p_lo = _adj(z_lo)
    p_hi = _adj(z_hi)

    lo = float(np.nanpercentile(boot_dist, 100 * p_lo))
    hi = float(np.nanpercentile(boot_dist, 100 * p_hi))
    return lo, hi
