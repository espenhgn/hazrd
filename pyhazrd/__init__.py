"""pyhazrd – Polygenic Hazard Score evaluation and visualization.

Primary entry points
--------------------
phs_metrics
    Compute discrimination metrics (HR, C-index, HR-per-SD, OR) for a
    polygenic hazard score, with optional bootstrapped CIs.
phs_km_curve
    Kaplan-Meier survival curves stratified by PHS percentile group.

Example
-------
>>> from pyhazrd import phs_metrics, phs_km_curve
>>> from pyhazrd.data import load_test_data
>>> data = load_test_data()
>>> result = phs_metrics(data, metrics=["HR", "C_index"])
>>> result[["metric", "estimate"]]
"""

from pyhazrd.metrics import phs_metrics
from pyhazrd.km import phs_km_curve
from pyhazrd.cox import phs_cox_curve

__version__ = "0.1.0"
__all__ = ["phs_metrics", "phs_km_curve", "phs_cox_curve"]
