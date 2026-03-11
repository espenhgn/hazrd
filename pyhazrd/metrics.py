"""Primary entry point for computing Polygenic Hazard Score metrics.

Example
-------
>>> import pandas as pd
>>> from pyhazrd import phs_metrics
>>> from pyhazrd.data import load_test_data
>>> data = load_test_data()
>>> phs_metrics(data, metrics=["HR", "C_index"])
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pyhazrd._metrics_internal import (
    _calc_cindex_metric,
    _calc_hr_metric,
    _calc_hrsd_metric,
    _calc_or_metric,
)


# ---------------------------------------------------------------------------
# phs_metrics
# ---------------------------------------------------------------------------


def phs_metrics(
    data: pd.DataFrame,
    phs: str = "phs",
    time: str = "age",
    event: str = "status",
    metrics: list[str] | None = None,
    hr_method: str = "continuous_group",
    hr_numerator: float | None = None,
    hr_denominator: float | None = None,
    hr_pairs: list[dict] | None = None,
    cindex_method: str = "harrell",
    or_age: float | list[float] | None = None,
    or_numerator: float | None = None,
    or_denominator: float | None = None,
    or_pairs: list[dict] | None = None,
    bootstrap: bool = False,
    n_boot: int = 999,
    ci_level: float = 0.95,
    boot_method: str = "percentile",
    seed: int | None = None,
    parallel: str = "no",
    n_cores: int = 1,
    strata: str | None = None,
) -> pd.DataFrame:
    """Calculate evaluation metrics for Polygenic Hazard Scores.

    Parameters
    ----------
    data:
        ``DataFrame`` containing columns specified by *phs*, *time*, and
        *event*.
    phs:
        Column name for the polygenic hazard score. Default ``"phs"``.
    time:
        Column name for time to event or censoring. Default ``"age"``.
    event:
        Column name for event indicator (0 = censored, 1 = event).
        Default ``"status"``.
    metrics:
        List of metric names to compute. Options are ``"HR"``,
        ``"C_index"``, ``"HR_SD"``, ``"OR"``. Default ``["HR", "C_index"]``.
    hr_method:
        Method for HR calculation. Currently only ``"continuous_group"``
        (default) and ``"continuous_point"`` are implemented.
    hr_numerator:
        Lower boundary of the numerator group as a percentile (e.g. ``0.80``
        for the top 20 %). Ignored when *hr_pairs* is provided.
        Default ``0.80``.
    hr_denominator:
        Upper boundary of the denominator group as a percentile (e.g.
        ``0.20`` for the bottom 20 %). Ignored when *hr_pairs* is provided.
        Default ``0.20``.
    hr_pairs:
        List of HR specs for multiple HRs. Each element must be a dict with
        keys ``"numerator"`` and ``"denominator"``, each a length-2 sequence
        of (lower, upper) percentile boundaries::

            hr_pairs = [
                {"numerator": (0.80, 1.00), "denominator": (0.00, 0.20)},
                {"numerator": (0.80, 1.00), "denominator": (0.40, 0.60)},
            ]

        Cannot be combined with *hr_numerator* / *hr_denominator*.
    cindex_method:
        ``"harrell"`` (default). ``"uno"`` is not yet implemented.
    or_age:
        Age(s) at which the odds ratio should be calculated. Required when
        ``"OR"`` is in *metrics*. May be a scalar or list.
    or_numerator:
        Lower boundary of the numerator band for OR. Default ``0.80``.
    or_denominator:
        Upper boundary of the denominator band for OR. Default ``0.20``.
    or_pairs:
        Same structure as *hr_pairs* but for OR calculation.
    bootstrap:
        Whether to compute bootstrapped confidence intervals.
        Default ``False``.
    n_boot:
        Number of bootstrap replicates. Default ``999``.
    ci_level:
        Confidence level for bootstrap CIs. Default ``0.95``.
    boot_method:
        One of ``"percentile"`` (default), ``"bca"``, or ``"normal"``.
    seed:
        Optional integer for reproducibility.
    parallel:
        Only ``"no"`` is currently implemented.
    n_cores:
        Reserved for future parallel support.
    strata:
        Optional column name to stratify bootstrap resampling on.

    Returns
    -------
    pandas.DataFrame
        One row per metric with columns:

        metric
            Full metric name, e.g. ``"HR[80-100]_[0-20]"``,
            ``"C_index"``, ``"OR[80-100]_[0-20]_age70"``.
        estimate
            Point estimate on the full dataset.
        conf_low
            Lower CI bound (``NaN`` when *bootstrap* is ``False``).
        conf_high
            Upper CI bound (``NaN`` when *bootstrap* is ``False``).
        se
            Bootstrap standard error (``NaN`` when *bootstrap* is ``False``).
        n_numerator
            Sample size in the numerator group (HR and OR only).
        n_denominator
            Sample size in the denominator group (HR and OR only).
        method
            Method flag, e.g. ``"continuous_group"``, ``"harrell"``.
        adjusted
            Always ``False`` until covariate support is added.
    """
    from lifelines import CoxPHFitter

    # ── default metrics ──────────────────────────────────────────────────────
    if metrics is None:
        metrics = ["HR", "C_index"]

    # ── input validation ─────────────────────────────────────────────────────
    for col in (phs, time, event):
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")

    valid_hr_methods = ("continuous_group", "continuous_point")
    if hr_method not in valid_hr_methods:
        raise ValueError(
            f"hr_method must be one of {valid_hr_methods}, got '{hr_method}'"
        )

    valid_cindex_methods = ("harrell", "uno")
    if cindex_method not in valid_cindex_methods:
        raise ValueError(
            f"cindex_method must be one of {valid_cindex_methods}, "
            f"got '{cindex_method}'"
        )

    valid_metrics = ("HR", "C_index", "HR_SD", "OR")
    bad = [m for m in metrics if m not in valid_metrics]
    if bad:
        raise ValueError(
            f"Unknown metric(s): {', '.join(bad)}. "
            f"Valid options are: {', '.join(valid_metrics)}."
        )

    valid_boot_methods = ("percentile", "bca", "normal")
    if boot_method not in valid_boot_methods:
        raise ValueError(
            f"boot_method must be one of {valid_boot_methods}, "
            f"got '{boot_method}'"
        )

    valid_parallel = ("no", "multicore", "snow")
    if parallel not in valid_parallel:
        raise ValueError(
            f"parallel must be one of {valid_parallel}, got '{parallel}'"
        )

    # ── resolve HR pairs ─────────────────────────────────────────────────────
    if "HR" in metrics:
        if hr_pairs is not None and (
            hr_numerator is not None or hr_denominator is not None
        ):
            raise ValueError(
                "Provide either hr_numerator/hr_denominator or hr_pairs, not both."
            )
        if hr_pairs is None:
            _hr_num = hr_numerator if hr_numerator is not None else 0.80
            _hr_den = hr_denominator if hr_denominator is not None else 0.20
            hr_pairs = [
                {"numerator": (_hr_num, 1.00), "denominator": (0.00, _hr_den)}
            ]
    else:
        hr_pairs = hr_pairs or []

    # ── resolve OR pairs ─────────────────────────────────────────────────────
    if "OR" in metrics:
        if or_age is None:
            raise ValueError("'or_age' is required when 'OR' is included in metrics.")
        if isinstance(or_age, (int, float)):
            or_age = [or_age]
        else:
            or_age = list(or_age)

        if or_pairs is not None and (
            or_numerator is not None or or_denominator is not None
        ):
            raise ValueError(
                "Provide either or_numerator/or_denominator or or_pairs, not both."
            )
        if or_pairs is None:
            _or_num = or_numerator if or_numerator is not None else 0.80
            _or_den = or_denominator if or_denominator is not None else 0.20
            or_pairs = [
                {"numerator": (_or_num, 1.00), "denominator": (0.00, _or_den)}
            ]
    else:
        or_pairs = or_pairs or []
        or_age = []

    # ── fit Cox model ────────────────────────────────────────────────────────
    needs_coxph = any(m in metrics for m in ("HR", "C_index", "HR_SD"))
    cxph = None
    if needs_coxph:
        cxph = CoxPHFitter()
        cxph.fit(data[[phs, time, event]], duration_col=time, event_col=event)

    # ── point estimates ───────────────────────────────────────────────────────
    parts: list[pd.DataFrame] = []

    if "HR" in metrics:
        for pair in hr_pairs:
            parts.append(
                _calc_hr_metric(
                    data, phs, time, event,
                    hr_method, pair["numerator"], pair["denominator"], cxph
                )
            )

    if "C_index" in metrics:
        parts.append(
            _calc_cindex_metric(data, phs, time, event, cindex_method, cxph)
        )

    if "HR_SD" in metrics:
        parts.append(_calc_hrsd_metric(data, phs, time, event, cxph))

    if "OR" in metrics:
        for pair in or_pairs:
            for age in or_age:
                parts.append(
                    _calc_or_metric(
                        data, phs, time, event,
                        pair["numerator"], pair["denominator"], age
                    )
                )

    point_estimates = pd.concat(parts, ignore_index=True)

    # ── bootstrap ────────────────────────────────────────────────────────────
    if not bootstrap:
        return point_estimates

    from pyhazrd._bootstrap_internal import _run_bootstrap

    return _run_bootstrap(
        point_estimates=point_estimates,
        data=data,
        phs=phs,
        time=time,
        event=event,
        metrics=metrics,
        hr_method=hr_method,
        hr_pairs=hr_pairs,
        cindex_method=cindex_method,
        or_age=or_age,
        or_pairs=or_pairs,
        n_boot=n_boot,
        ci_level=ci_level,
        boot_method=boot_method,
        seed=seed,
        parallel=parallel,
        n_cores=n_cores,
        strata_col=strata,
    )
