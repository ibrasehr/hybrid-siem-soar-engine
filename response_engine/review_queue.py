"""Analyst review queue with a hard capacity. Overflow is surfaced, never dropped."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .policy import TIER_REVIEW


@dataclass
class QueueResult:
    queue: pd.DataFrame        # what the analyst sees now (highest confidence first)
    overflow: pd.DataFrame     # waiting beyond capacity, with overflow_reason
    capacity: int

    @property
    def overflow_count(self) -> int:
        return len(self.overflow)


def build_review_queue(decisions: pd.DataFrame, capacity: int = 50) -> QueueResult:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError(f"capacity must be an int >= 1, got {capacity!r}")
    review = decisions[decisions["tier"] == TIER_REVIEW]
    ordered = review.sort_values(["confidence", "id"], ascending=[False, True], kind="mergesort")
    overflow = ordered.iloc[capacity:].copy()
    overflow["overflow_reason"] = "queue_capacity"
    return QueueResult(ordered.head(capacity).copy(), overflow, capacity)
