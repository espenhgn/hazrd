"""Sample datasets bundled with pyhazrd."""

from __future__ import annotations

import importlib.resources as _resources
from pathlib import Path

import pandas as pd


def load_test_data() -> pd.DataFrame:
    """Return the bundled ``test_data`` DataFrame.

    This is a simulated dataset of 1 000 individuals with columns:

    phs
        Polygenic hazard score (continuous).
    age
        Age at diagnosis or age at censoring.
    status
        Event indicator: 1 = case, 0 = censored.

    Returns
    -------
    pandas.DataFrame
    """
    try:
        ref = _resources.files("pyhazrd.data") / "test_data.csv"
        with _resources.as_file(ref) as path:
            return pd.read_csv(path)
    except (AttributeError, TypeError):
        # Python < 3.9 fallback
        data_dir = Path(__file__).parent
        return pd.read_csv(data_dir / "test_data.csv")
