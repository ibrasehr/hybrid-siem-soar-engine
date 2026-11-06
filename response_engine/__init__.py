"""Governed SOAR response engine.

    from response_engine import Policy, run_pipeline, load_alerts
    result = run_pipeline(load_alerts("alerts.json"), Policy(auto_threshold=0.9))
"""
__version__ = "2.0.0"

from .adapters import load_alerts, load_insider_csv
from .assets import AssetRegistry, bdi
from .catalog import CATALOG, SEVERITY, Action, UnknownAction, action_for, find_action
from .engine import audit_jsonl, audit_records, decide_all, explain
from .feedback import OUTCOMES, FeedbackLog, suggest_threshold
from .ledger import (AlreadyContained, AlreadyReverted, Ledger, LedgerError, StateDrift,
                     UnknownIncident)
from .pipeline import PipelineResult, run_pipeline
from .policy import Policy
from .review_queue import QueueResult, build_review_queue

__all__ = [
    "Policy", "decide_all", "explain", "audit_records", "audit_jsonl", "run_pipeline",
    "PipelineResult", "Ledger", "LedgerError", "AlreadyContained", "AlreadyReverted",
    "UnknownIncident", "StateDrift", "FeedbackLog", "suggest_threshold", "OUTCOMES",
    "build_review_queue", "QueueResult", "AssetRegistry", "bdi", "CATALOG", "SEVERITY",
    "Action", "UnknownAction", "action_for", "find_action", "load_alerts", "load_insider_csv",
]
