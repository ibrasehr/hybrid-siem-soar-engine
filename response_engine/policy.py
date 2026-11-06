"""Policy: every tunable number in one validated, immutable object."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Optional

TIER_LOG, TIER_REVIEW, TIER_AUTO_SOFT, TIER_AUTO_HARD = 0, 1, 2, 3
TIER_NAMES = {TIER_LOG: "LOG_ONLY", TIER_REVIEW: "REVIEW",
              TIER_AUTO_SOFT: "AUTO_SOFT", TIER_AUTO_HARD: "AUTO_HARD"}
DECISION_LABELS = {TIER_LOG: "LOG ONLY", TIER_REVIEW: "HUMAN REVIEW",
                   TIER_AUTO_SOFT: "AUTO-RESPONSE", TIER_AUTO_HARD: "AUTO-RESPONSE"}


@dataclass(frozen=True)
class Policy:
    """review_floor   : below this confidence an alert is only logged
    auto_threshold : at/above this, the engine may act without a human
    hard_threshold : at/above this, the action is hard containment (else soft)
    bdi_limit      : max Business Disruption Index allowed without a human
                     (None disables the gate)
    queue_capacity : alerts one analyst queue can hold at once
    """
    review_floor: float = 0.50
    auto_threshold: float = 0.90
    hard_threshold: float = 0.97
    bdi_limit: Optional[float] = 12.0
    queue_capacity: int = 50

    def __post_init__(self):
        for name in ("review_floor", "auto_threshold", "hard_threshold"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be a number in [0, 1], got {v!r}")
        if not self.review_floor <= self.auto_threshold <= self.hard_threshold:
            raise ValueError("need review_floor <= auto_threshold <= hard_threshold")
        if self.bdi_limit is not None and not self.bdi_limit > 0:
            raise ValueError(f"bdi_limit must be > 0 or None, got {self.bdi_limit!r}")
        if isinstance(self.queue_capacity, bool) or not isinstance(self.queue_capacity, int) \
                or self.queue_capacity < 1:
            raise ValueError(f"queue_capacity must be an int >= 1, got {self.queue_capacity!r}")

    def with_auto_threshold(self, t: float) -> "Policy":
        """Move the auto threshold while keeping the ordering invariant."""
        return replace(self, review_floor=min(self.review_floor, t), auto_threshold=t,
                       hard_threshold=max(self.hard_threshold, t))

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown policy keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)
