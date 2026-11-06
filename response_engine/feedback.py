"""Feedback log: confirmed outcomes written back so thresholds can be recalibrated.

This builds the PIPE for a closed loop. suggest_threshold() only SUGGESTS a value;
nothing is ever applied automatically.
"""
from __future__ import annotations

import csv
import datetime as _dt
import os
from typing import Callable, Optional

import pandas as pd

OUTCOMES = ("analyst_approved", "analyst_rejected", "reverted_false_positive")
POSITIVE = {"analyst_approved"}
COLUMNS = ["timestamp", "incident_id", "source", "original_confidence", "decision",
           "outcome", "reviewer"]


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class FeedbackLog:
    def __init__(self, path: Optional[str] = None, clock: Optional[Callable[[], str]] = None):
        self.path, self._clock, self._rows = path, clock or _utcnow, []

    def record(self, incident_id, source, confidence, decision, outcome, reviewer="analyst"):
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        row = {"timestamp": self._clock(), "incident_id": incident_id, "source": source,
               "original_confidence": float(confidence), "decision": decision,
               "outcome": outcome, "reviewer": reviewer}
        self._rows.append(row)
        if self.path:
            new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
            with open(self.path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                if new:
                    w.writeheader()
                w.writerow(row)
        return row

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=COLUMNS)

    def __len__(self):
        return len(self._rows)


def suggest_threshold(feedback: pd.DataFrame, target_precision: float = 0.95, min_n: int = 20) -> dict:
    """Lowest auto threshold whose confirmed-outcome precision meets the target.

    approved = positive; rejected / reverted false positive = negative.
    Returns a suggestion only (applied is always False).
    """
    base = {"suggested_threshold": None, "applied": False, "support": 0, "reason": ""}
    if feedback is None or len(feedback) == 0:
        return {**base, "reason": "no feedback recorded"}
    df = feedback.copy()
    df["positive"] = df["outcome"].isin(POSITIVE)
    best = None
    for t in sorted(df["original_confidence"].unique()):
        sub = df[df["original_confidence"] >= t]
        if len(sub) >= min_n and sub["positive"].mean() >= target_precision:
            best = (float(t), len(sub))
            break
    if best is None:
        return {**base, "support": len(df),
                "reason": f"no threshold reaches precision {target_precision:.2f} with n >= {min_n}"}
    return {**base, "suggested_threshold": best[0], "support": best[1],
            "reason": f"precision >= {target_precision:.2f} on {best[1]} confirmed outcomes"}
