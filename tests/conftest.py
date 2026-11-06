import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_df(confs, source="correlation", labels=None, prefix="INC"):
    n = len(confs)
    return pd.DataFrame({
        "id": [f"{prefix}-{i}" for i in range(n)],
        "target": [f"flow-{i}" for i in range(n)],
        "source": [source] * n,
        "confidence": list(confs),
        "true_label": list(labels) if labels is not None else [1] * n,
        "family": ["x"] * n,
        "rules_fired": [[] for _ in range(n)],
        "top_features": [[] for _ in range(n)],
    })


@pytest.fixture
def mk():
    return make_df


@pytest.fixture
def clock():
    """Deterministic clock: t0001, t0002, ..."""
    n = {"i": 0}

    def tick():
        n["i"] += 1
        return f"t{n['i']:04d}"
    return tick
