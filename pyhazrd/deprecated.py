"""Deprecated compatibility wrappers.

These functions mirror the pre-0.2.0 API of the hazrd R package.  They emit a
:class:`DeprecationWarning` and forward to the modern :func:`phs_metrics`
interface.

Removal target: 0.3.0
"""

from __future__ import annotations

import warnings

from pyhazrd.metrics import phs_metrics


def get_hr(
    data=None,
    phs: str = "phs",
    age: str = "age",
    status: str = "status",
    lower_interval=0.20,
    upper_interval=0.80,
    CI: bool = False,
    bootstrap_iterations: int = 1000,
    swc: bool = False,
    swc_popnumcases=None,
    swc_popnumcontrols=None,
) -> dict:
    """Deprecated: use :func:`phs_metrics` instead.

    .. deprecated::
        Use ``phs_metrics(data, metrics=["HR"], ...)`` instead.
    """
    # Expand single-value intervals to bands
    if hasattr(lower_interval, "__len__") and len(lower_interval) == 2:
        denominator = tuple(lower_interval)
    else:
        denominator = (0.0, float(lower_interval))

    if hasattr(upper_interval, "__len__") and len(upper_interval) == 2:
        numerator = tuple(upper_interval)
    else:
        numerator = (float(upper_interval), 1.0)

    if swc:
        warnings.warn(
            "Sample weight correction (swc) is not yet implemented in phs_metrics(). "
            "The swc argument will be ignored.",
            DeprecationWarning,
            stacklevel=2,
        )

    warnings.warn(
        f"'get_hr()' is deprecated. "
        f"Use phs_metrics(data, metrics=['HR'], hr_pairs=[{{'numerator': {numerator}, "
        f"'denominator': {denominator}}}]) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    result = phs_metrics(
        data=data,
        phs=phs,
        time=age,
        event=status,
        metrics=["HR"],
        hr_pairs=[{"numerator": numerator, "denominator": denominator}],
        bootstrap=CI,
        n_boot=bootstrap_iterations,
        ci_level=0.95,
    )

    upper_val = upper_interval[-1] if hasattr(upper_interval, "__len__") else upper_interval
    lower_val = lower_interval[0] if hasattr(lower_interval, "__len__") else lower_interval

    return {
        "index": f"HR{round(upper_val * 100)}_{round(lower_val * 100)}",
        "value": float(result["estimate"].iloc[0]),
        "conf.low": result["conf_low"].iloc[0],
        "conf.high": result["conf_high"].iloc[0],
        "iters": bootstrap_iterations if CI else None,
    }


def get_cindex(
    data=None,
    phs: str = "phs",
    age: str = "age",
    status: str = "status",
    bootstrap_iterations: int | None = None,
    conf_level: float = 0.95,
) -> dict:
    """Deprecated: use :func:`phs_metrics` instead.

    .. deprecated::
        Use ``phs_metrics(data, metrics=["C_index"])`` instead.
    """
    warnings.warn(
        "'get_cindex()' is deprecated. "
        "Use phs_metrics(data, metrics=['C_index']) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    do_boot = bootstrap_iterations is not None

    result = phs_metrics(
        data=data,
        phs=phs,
        time=age,
        event=status,
        metrics=["C_index"],
        bootstrap=do_boot,
        n_boot=bootstrap_iterations if do_boot else 1000,
        ci_level=conf_level,
    )

    out = {"value": float(result["estimate"].iloc[0])}
    if do_boot:
        out["conf.low"] = result["conf_low"].iloc[0]
        out["conf.high"] = result["conf_high"].iloc[0]
    return out


def get_hrsd(
    data=None,
    phs: str = "phs",
    age: str = "age",
    status: str = "status",
    conf_int: bool = False,
    conf_level: float = 0.95,
    bootstrap_iterations: int = 1000,
) -> dict:
    """Deprecated: use :func:`phs_metrics` instead.

    .. deprecated::
        Use ``phs_metrics(data, metrics=["HR_SD"])`` instead.
    """
    warnings.warn(
        "'get_hrsd()' is deprecated. "
        "Use phs_metrics(data, metrics=['HR_SD']) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    result = phs_metrics(
        data=data,
        phs=phs,
        time=age,
        event=status,
        metrics=["HR_SD"],
        bootstrap=conf_int,
        n_boot=bootstrap_iterations,
        ci_level=conf_level,
    )

    out = {"value": float(result["estimate"].iloc[0])}
    if conf_int:
        out["conf.low"] = result["conf_low"].iloc[0]
        out["conf.high"] = result["conf_high"].iloc[0]
    return out


def get_or(
    data=None,
    phs: str = "phs",
    age: str = "age",
    status: str = "status",
    or_age=None,
    numerator=(0.8, 1.0),
    denominator=(0.0, 0.2),
    bootstrap_iterations: int | None = None,
    conf_level: float = 0.95,
) -> dict:
    """Deprecated: use :func:`phs_metrics` instead.

    .. deprecated::
        Use ``phs_metrics(data, metrics=["OR"], or_age=...)`` instead.
    """
    if or_age is None:
        raise ValueError(
            "Argument 'or_age' is required. Please specify the age at which "
            "to compute the OR."
        )

    warnings.warn(
        f"'get_or()' is deprecated. "
        f"Use phs_metrics(data, metrics=['OR'], or_age={or_age}) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    do_boot = bootstrap_iterations is not None

    result = phs_metrics(
        data=data,
        phs=phs,
        time=age,
        event=status,
        metrics=["OR"],
        or_age=or_age,
        or_pairs=[{"numerator": tuple(numerator), "denominator": tuple(denominator)}],
        bootstrap=do_boot,
        n_boot=bootstrap_iterations if do_boot else 1000,
        ci_level=conf_level,
    )

    out = {"value": float(result["estimate"].iloc[0])}
    if do_boot:
        out["conf.low"] = result["conf_low"].iloc[0]
        out["conf.high"] = result["conf_high"].iloc[0]
    return out


def km_curve(
    data=None,
    phs: str = "phs",
    age: str = "age",
    status: str = "status",
    interval=(0, 1),
    age_range=None,
    scale: bool = False,
    inverse: bool = False,
) -> object:
    """Deprecated: use :func:`phs_km_curve` instead.

    .. deprecated::
        Use :func:`phs_km_curve` instead.
    """
    from pyhazrd.km import phs_km_curve

    if scale:
        warnings.warn(
            "The 'scale' argument is not supported in phs_km_curve(). "
            "It will be ignored. Scale your PHS column manually before calling "
            "phs_km_curve().",
            DeprecationWarning,
            stacklevel=2,
        )
    if inverse:
        warnings.warn(
            "The 'inverse' argument is not supported in phs_km_curve(). "
            "It will be ignored. Multiply your PHS column by -1 before calling "
            "phs_km_curve().",
            DeprecationWarning,
            stacklevel=2,
        )
    if age_range is not None:
        warnings.warn(
            "The 'age_range' argument is not supported in phs_km_curve(). "
            "It will be ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
    if list(interval) != [0, 1]:
        warnings.warn(
            "The 'interval' argument is not directly supported in phs_km_curve(). "
            "It will be ignored; all subjects are included.",
            DeprecationWarning,
            stacklevel=2,
        )

    warnings.warn(
        f"'km_curve()' is deprecated. "
        f"Use phs_km_curve(data, phs='{phs}', time='{age}', "
        f"event='{status}') instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return phs_km_curve(
        data=data,
        phs=phs,
        time=age,
        event=status,
        output="data",
    )
