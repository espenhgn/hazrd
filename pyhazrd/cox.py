"""Cox model survival curves at specified PHS percentiles."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Numerical floor to avoid log(0) when computing log-log CIs
_EPS = 1e-10


def phs_cox_curve(
    data: pd.DataFrame,
    phs: str = "phs",
    time: str = "age",
    event: str = "status",
    percentiles: list[float] | None = None,
    ref_data: pd.DataFrame | None = None,
    output: str = "plot",
    conf_int: bool = True,
    conf_int_alpha: float = 0.15,
    palette: str = "hazrd",
) -> object:
    """Cox model survival curves at specified PHS percentiles.

    Fit a Cox proportional-hazards model with *phs* as the sole predictor and
    return predicted survival curves for individuals at specified PHS
    percentiles. Unlike :func:`phs_km_curve`, these are smooth model-based
    predictions rather than empirical group estimates.

    Uses :class:`sksurv.linear_model.CoxPHSurvivalAnalysis` for fitting and
    prediction. When ``conf_int=True``, a second fit via
    :class:`lifelines.CoxPHFitter` is performed to obtain the coefficient
    variance matrix (scikit-survival does not expose this directly); both fits
    converge to the same estimate.

    Parameters
    ----------
    data:
        ``DataFrame`` with columns specified by *phs*, *time*, and *event*.
    phs:
        Column name for the polygenic hazard score. Default ``"phs"``.
    time:
        Column name for time to event or censoring. Default ``"age"``.
    event:
        Column name for event indicator (0 = censored, 1 = event).
        Default ``"status"``.
    percentiles:
        Percentile values strictly in (0, 1) at which to compute Cox-predicted
        survival curves. Default ``[0.01, 0.05, 0.20, 0.50, 0.80, 0.95, 0.99]``.
    ref_data:
        Optional reference ``DataFrame`` used to compute the PHS value at each
        requested percentile (training reference). When ``None``, percentiles
        are computed from *data*.
    output:
        ``"plot"`` (default) returns a :class:`matplotlib.figure.Figure`;
        ``"data"`` returns a ``DataFrame``.
    conf_int:
        Include confidence intervals in output/plot. Default ``True``.
    conf_int_alpha:
        Alpha for confidence ribbons in plots. Default ``0.15``.
    palette:
        Colour palette for the plot. Default ``"hazrd"`` uses a built-in
        palette derived from ``RdYlBu``.

    Returns
    -------
    matplotlib.figure.Figure or pandas.DataFrame
        Depending on the *output* argument. The ``DataFrame`` has columns
        ``time``, ``estimate``, ``conf.low``, ``conf.high``, ``percentile``,
        and ``percentile_value``.
    """
    from scipy.stats import norm
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    if percentiles is None:
        percentiles = [0.01, 0.05, 0.20, 0.50, 0.80, 0.95, 0.99]

    # ── input validation ──────────────────────────────────────────────────────
    if not isinstance(data, pd.DataFrame):
        raise TypeError("'data' must be a pandas DataFrame.")
    for col in (phs, time, event):
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    if output not in ("plot", "data"):
        raise ValueError("'output' must be 'plot' or 'data'.")
    percentiles_arr = np.asarray(percentiles, dtype=float)
    if not (np.all(percentiles_arr > 0) and np.all(percentiles_arr < 1)):
        raise ValueError(
            "'percentiles' must be a list with all values strictly between 0 and 1."
        )

    # ── resolve column vectors ────────────────────────────────────────────────
    phs_vals = data[phs].to_numpy(dtype=float)
    time_vals = data[time].to_numpy(dtype=float)
    event_vals = data[event].to_numpy(dtype=bool)

    # ── fit Cox model via scikit-survival ─────────────────────────────────────
    y = np.array(
        list(zip(event_vals, time_vals)),
        dtype=[("event", bool), ("time", float)],
    )
    X = phs_vals.reshape(-1, 1)

    cox = CoxPHSurvivalAnalysis(ties="breslow")
    cox.fit(X, y)

    # ── coefficient SE for Wald CIs ───────────────────────────────────────────
    # scikit-survival does not expose the covariance matrix; a second fit via
    # lifelines (which shares the same partial-likelihood maximisation) is used
    # solely to obtain var(beta) for the log-log CI transform.
    beta_se: float | None = None
    if conf_int:
        from lifelines import CoxPHFitter

        cxph = CoxPHFitter()
        cxph.fit(
            data[[phs, time, event]],
            duration_col=time,
            event_col=event,
        )
        beta_se = float(np.sqrt(cxph.variance_matrix_.values[0, 0]))

    # ── PHS values at requested percentiles ───────────────────────────────────
    ref_phs = (
        ref_data[phs].to_numpy(dtype=float)
        if ref_data is not None
        else phs_vals
    )
    phs_at_pct = np.quantile(ref_phs, percentiles_arr)
    pct_labels = [f"P{round(p * 100)}" for p in percentiles_arr]

    # ── predict survival curves via scikit-survival ───────────────────────────
    X_new = phs_at_pct.reshape(-1, 1)
    sf_fns = cox.predict_survival_function(X_new, return_array=False)

    z_crit = norm.ppf(0.975)

    out_frames: list[pd.DataFrame] = []
    for sf_fn, label, pct_val, phs_val in zip(
        sf_fns, pct_labels, percentiles_arr, phs_at_pct
    ):
        times = sf_fn.x
        surv = np.clip(sf_fn(times), 0.0, 1.0)

        # Wald CI on log-log scale: log(-log S(t)) +/- z * |x| * se(beta)
        if conf_int and beta_se is not None:
            log_neg_log_s = np.log(-np.log(np.clip(surv, _EPS, 1 - _EPS)))
            half_width = z_crit * abs(phs_val) * beta_se
            conf_lo = np.clip(
                np.exp(-np.exp(log_neg_log_s + half_width)), 0.0, 1.0
            )
            conf_hi = np.clip(
                np.exp(-np.exp(log_neg_log_s - half_width)), 0.0, 1.0
            )
        else:
            conf_lo = np.full_like(surv, np.nan)
            conf_hi = np.full_like(surv, np.nan)

        out_frames.append(
            pd.DataFrame(
                {
                    "time": times,
                    "estimate": surv,
                    "conf.low": conf_lo,
                    "conf.high": conf_hi,
                    "percentile": label,
                    "percentile_value": float(pct_val),
                }
            )
        )

    out_df = pd.concat(out_frames, ignore_index=True)

    # Preserve label order (low -> high percentile)
    out_df["percentile"] = pd.Categorical(
        out_df["percentile"], categories=pct_labels, ordered=True
    )

    if output == "data":
        return out_df

    # ── build plot ────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    n = len(pct_labels)
    cmap = plt.colormaps["RdYlBu"].resampled(n)
    color_map = {lbl: cmap(i / max(n - 1, 1)) for i, lbl in enumerate(pct_labels)}

    for label in pct_labels:
        sub = out_df[out_df["percentile"] == label]
        if sub.empty:
            continue
        color = color_map[label]
        ax.plot(
            sub["time"], sub["estimate"],
            color=color, label=label, linewidth=0.8,
        )
        if conf_int:
            ax.fill_between(
                sub["time"],
                sub["conf.low"],
                sub["conf.high"],
                alpha=conf_int_alpha,
                color=color,
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival")
    ax.legend(title="Percentile")

    return fig
