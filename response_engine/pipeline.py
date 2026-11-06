"""One-call pipeline: decide -> execute autos -> queue reviews -> apply analyst actions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import pandas as pd

from .assets import AssetRegistry
from .catalog import find_action
from .engine import decide_all
from .feedback import FeedbackLog
from .ledger import Ledger
from .policy import TIER_REVIEW, Policy
from .review_queue import QueueResult, build_review_queue


@dataclass
class PipelineResult:
    decisions: pd.DataFrame
    ledger: Ledger
    queue: QueueResult
    feedback: FeedbackLog


def run_pipeline(df: pd.DataFrame, policy: Optional[Policy] = None,
                 registry: Optional[AssetRegistry] = None,
                 approve_ids: Iterable[str] = (), reject_ids: Iterable[str] = (),
                 revert_ids: Iterable[str] = (), clock: Optional[Callable[[], str]] = None,
                 feedback_path: Optional[str] = None,
                 with_explanations: bool = True) -> PipelineResult:
    policy = policy or Policy()
    dec = decide_all(df, policy=policy, registry=registry, with_explanations=with_explanations)
    ledger = Ledger(clock=clock)
    feedback = FeedbackLog(path=feedback_path, clock=clock)

    for r in dec[dec["tier"] >= 2].itertuples():                 # AUTO_SOFT / AUTO_HARD
        ledger.execute(r.id, r.target, r.source, r.action, r.confidence, "auto")

    by_id = dec.set_index("id", drop=False)
    for i in approve_ids:
        row = _review_row(by_id, i)
        find_action(row["source"], row["proposed_action"])        # must be an approved action
        ledger.execute(i, row["target"], row["source"], row["proposed_action"],
                       row["confidence"], "analyst")
        feedback.record(i, row["source"], row["confidence"], row["decision"], "analyst_approved")
    for i in reject_ids:
        row = _review_row(by_id, i)
        feedback.record(i, row["source"], row["confidence"], row["decision"], "analyst_rejected")
    for i in revert_ids:
        entry = ledger.revert(i)                                  # UnknownIncident / StateDrift ...
        row = by_id.loc[i]
        feedback.record(i, entry.source, entry.confidence, row["decision"], "reverted_false_positive")

    return PipelineResult(dec, ledger, build_review_queue(dec, policy.queue_capacity), feedback)


def _review_row(by_id: pd.DataFrame, incident_id: str) -> dict:
    if incident_id not in by_id.index:
        raise ValueError(f"unknown incident {incident_id!r}")
    row = by_id.loc[incident_id]
    if row["tier"] != TIER_REVIEW:
        raise ValueError(f"{incident_id} is not in the review queue (tier {row['tier']})")
    return row.to_dict()
