"""Asset criticality (1-5) and the Business Disruption Index.

BDI = asset criticality x action severity   (range 2..25 for real actions)

UNSW-NB15 flows carry no host/IP, so by default criticality is SIMULATED with a
stable hash of the target. Pass a real registry (AssetRegistry.from_csv) to
replace it. `registry.simulated` tells you which one you are using; report it.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from .catalog import SEVERITY


def bdi(criticality: int, kind: str) -> int:
    return int(criticality) * SEVERITY[kind]


def _validate(c):
    if isinstance(c, bool) or not isinstance(c, (int, float)) or int(c) != c or not 1 <= c <= 5:
        raise ValueError(f"criticality must be an integer 1-5, got {c!r}")
    return int(c)


class AssetRegistry:
    def __init__(self, mapping: dict | None = None, default: int = 3):
        self._mapping = None if mapping is None else {k: _validate(v) for k, v in mapping.items()}
        self._default = _validate(default)

    @property
    def simulated(self) -> bool:
        return self._mapping is None

    @classmethod
    def from_csv(cls, path, default: int = 3) -> "AssetRegistry":
        df = pd.read_csv(path)
        if not {"target", "criticality"} <= set(df.columns):
            raise ValueError("asset CSV needs columns: target, criticality")
        return cls(dict(zip(df["target"], df["criticality"])), default=default)

    def criticality(self, target: str) -> int:
        if self._mapping is not None:
            return self._mapping.get(target, self._default)
        bucket = int(hashlib.md5(str(target).encode()).hexdigest(), 16) % 100   # stable across runs
        return 1 if bucket < 30 else 2 if bucket < 60 else 3 if bucket < 80 else 4 if bucket < 92 else 5
