"""Kaplan-Meier curves stratified by PHS percentile group."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def phs_km_curve(
    data: pd.DataFrame,
    phs: str = "phs",
    time: str = "age",
    event: str = "status",
    breaks: list[float] | None = None,
    ref_data: pd.DataFrame | None = None,
    output: str = "plot",
    conf_int: bool = True,
    conf_int_alpha: float = 0.15,
    palette: str = "hazrd",
    risk_table: bool = False,
) -> object:
    """Kaplan-Meier curves stratified by PHS percentile.

    Parameters
    ----------
    data:
        ``DataFrame`` with columns specified by *phs*, *time*, *event*.
    phs:
        Column name for PHS values. Default ``"phs"``.
    time:
        Column name for event time. Default ``"age"``.
    event:
        Column name for event indicator (0/1). Default ``"status"``.
    breaks:
        Percentile cutpoints strictly in (0, 1). Default ``[0.20, 0.80]``
        (bottom 20 % / middle 60 % / top 20 %).
    ref_data:
        Optional reference ``DataFrame`` used to compute percentile
        cutpoints (training reference).
    output:
        ``"plot"`` (default) returns a :class:`matplotlib.figure.Figure`;
        ``"data"`` returns a ``DataFrame``.
    conf_int:
        Include confidence intervals. Default ``True``.
    conf_int_alpha:
        Alpha for confidence ribbons in plots. Default ``0.15``.
    palette:
        Colour palette for the plot. Default ``"hazrd"`` uses a built-in
        palette. Any value accepted by :func:`matplotlib.pyplot.rcParams`
        can also be used.
    risk_table:
        When ``True`` and ``output="data"``, the returned ``DataFrame``
        includes ``n.risk`` and ``n.event`` columns. Default ``False``.

    Returns
    -------
    matplotlib.figure.Figure or pandas.DataFrame
        Depending on the *output* argument.
    """
    from lifelines import KaplanMeierFitter

    if breaks is None:
        breaks = [0.20, 0.80]

    # ── input validation ─────────────────────────────────────────────────────
    if not isinstance(data, pd.DataFrame):
        raise TypeError("'data' must be a pandas DataFrame.")
    for col in (phs, time, event):
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    if output not in ("plot", "data"):
        raise ValueError("'output' must be 'plot' or 'data'.")
    breaks_arr = np.array(breaks, dtype=float)
    if not (np.all(breaks_arr > 0) and np.all(breaks_arr < 1)):
        raise ValueError(
            "'breaks' must be a list with all values strictly between 0 and 1."
        )

    # ── resolve column vectors ────────────────────────────────────────────────
    phs_vals = data[phs].to_numpy(dtype=float)
    time_vals = data[time].to_numpy(dtype=float)
    event_vals = data[event].to_numpy(dtype=int)

    # ── percentile cutpoints from reference data ─────────────────────────────
    if ref_data is not None:
        ref_phs = ref_data[phs].to_numpy(dtype=float)
    else:
        ref_phs = phs_vals

    from scipy.stats import rankdata

    phs_pct = rankdata(phs_vals, method="average") / len(phs_vals)
    if ref_data is not None:
        # Compute ECDF of ref_phs and evaluate at each phs_val
        ref_sorted = np.sort(ref_phs)

        def _ecdf(x: float) -> float:
            return float(np.searchsorted(ref_sorted, x, side="right")) / len(
                ref_sorted
            )

        phs_pct = np.array([_ecdf(v) for v in phs_vals])

    # ── stratum labels ────────────────────────────────────────────────────────
    cuts = np.sort(np.unique(breaks_arr))
    cut_bounds = np.concatenate([[0.0], cuts, [1.0]])
    labels = [
        f"{round(cut_bounds[i]*100)}-{round(cut_bounds[i+1]*100)}%"
        for i in range(len(cut_bounds) - 1)
    ]

    stratum = np.empty(len(phs_pct), dtype=object)
    for i in range(len(cut_bounds) - 1):
        lo = cut_bounds[i]
        hi = cut_bounds[i + 1]
        if i == len(cut_bounds) - 2:  # last band: include right edge
            mask = (phs_pct >= lo) & (phs_pct <= hi)
        else:
            mask = (phs_pct >= lo) & (phs_pct < hi)
        stratum[mask] = labels[i]

    # ── fit KM per stratum ────────────────────────────────────────────────────
    out_frames: list[pd.DataFrame] = []
    for label in labels:
        mask = stratum == label
        if not mask.any():
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(time_vals[mask], event_observed=event_vals[mask])

        sf = kmf.survival_function_.reset_index()
        sf.columns = ["time", "estimate"]

        if conf_int:
            ci = kmf.confidence_interval_.reset_index()
            # columns: timeline, KM_estimate_lower_X, KM_estimate_upper_X
            sf["conf.low"] = ci.iloc[:, 1].to_numpy()
            sf["conf.high"] = ci.iloc[:, 2].to_numpy()
        else:
            sf["conf.low"] = np.nan
            sf["conf.high"] = np.nan

        if risk_table:
            ev = kmf.event_table_
            sf["n.risk"] = np.interp(
                sf["time"].to_numpy(),
                ev.index.to_numpy(),
                ev["at_risk"].to_numpy(),
            )
            sf["n.event"] = np.interp(
                sf["time"].to_numpy(),
                ev.index.to_numpy(),
                ev["observed"].to_numpy(),
            )

        sf["stratum"] = label
        out_frames.append(sf)

    out_df = pd.concat(out_frames, ignore_index=True)

    # Clamp to [0, 1]
    for col in ("estimate", "conf.low", "conf.high"):
        if col in out_df.columns:
            out_df[col] = out_df[col].clip(0.0, 1.0)

    if output == "data":
        return out_df

    # ── build plot ────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # Default hazrd palette (Set1-like colors)
    _hazrd_colors = [
        "#E41A1C",
        "#377EB8",
        "#4DAF4A",
        "#984EA3",
        "#FF7F00",
        "#A65628",
        "#F781BF",
        "#999999",
    ]

    color_map = {lbl: _hazrd_colors[i % len(_hazrd_colors)] for i, lbl in enumerate(labels)}

    for label in labels:
        sub = out_df[out_df["stratum"] == label]
        if sub.empty:
            continue
        color = color_map[label]
        ax.step(sub["time"], sub["estimate"], where="post", color=color, label=label, linewidth=0.8)
        if conf_int:
            ax.fill_between(
                sub["time"],
                sub["conf.low"],
                sub["conf.high"],
                step="post",
                alpha=conf_int_alpha,
                color=color,
            )

    ax.set_xlabel("Time")
    ax.set_ylabel("Survival")
    ax.legend(title="Stratum")

    return fig
