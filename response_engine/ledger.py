"""Transactional rollback ledger with a real state model.

Every executed action stores a snapshot of the target's prior state and the
exact inverse action. revert() restores the snapshot and VERIFIES it. If the
target's state changed behind our back, revert refuses (StateDrift) instead of
guessing -- that is the "stuck containment" failure the literature describes.
One active containment per target at a time.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import pandas as pd

from .catalog import find_action

NORMAL = "normal"
ACTIVE, REVERTED = "ACTIVE", "REVERTED"


class LedgerError(Exception): ...
class AlreadyContained(LedgerError): ...
class AlreadyReverted(LedgerError): ...
class UnknownIncident(LedgerError): ...
class StateDrift(LedgerError): ...


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class Entry:
    incident_id: str
    target: str
    source: str
    action: str
    undo_action: str
    prior_state: str
    applied_state: str
    confidence: float
    approved_by: str            # "auto" or "analyst"
    executed_at: str
    status: str = ACTIVE
    reverted_at: str = ""
    verified: Optional[bool] = None


COLUMNS = ["incident_id", "target", "source", "action", "undo_action", "prior_state",
           "applied_state", "confidence", "approved_by", "executed_at", "status",
           "reverted_at", "verified"]


class Ledger:
    def __init__(self, clock: Optional[Callable[[], str]] = None, initial_state: Optional[dict] = None):
        self._clock = clock or _utcnow
        self._state: dict = dict(initial_state or {})
        self._entries: dict = {}

    # ---- state ----
    def state_of(self, target: str) -> str:
        return self._state.get(target, NORMAL)

    # ---- execute / revert ----
    def execute(self, incident_id, target, source, action, confidence, approved_by) -> Entry:
        existing = self._entries.get(incident_id)
        if existing is not None:
            if existing.status == ACTIVE:
                return existing                                  # idempotent re-execution
            raise AlreadyReverted(f"{incident_id} was already reverted; it cannot be re-executed")
        act = find_action(source, action)                        # UnknownAction if not approved
        if self.state_of(target) != NORMAL:
            raise AlreadyContained(f"{target} is already in state {self.state_of(target)!r}")
        entry = Entry(incident_id, target, source, act.name, act.undo, self.state_of(target),
                      act.state, float(confidence), approved_by, self._clock())
        self._state[target] = act.state
        self._entries[incident_id] = entry
        return entry

    def revert(self, incident_id) -> Entry:
        e = self._entries.get(incident_id)
        if e is None:
            raise UnknownIncident(incident_id)
        if e.status == REVERTED:
            raise AlreadyReverted(incident_id)
        if self.state_of(e.target) != e.applied_state:
            raise StateDrift(f"{e.target}: expected {e.applied_state!r} but found "
                             f"{self.state_of(e.target)!r}; manual intervention required")
        self._state[e.target] = e.prior_state                    # inverse action restores snapshot
        e.verified = self.state_of(e.target) == e.prior_state
        if not e.verified:
            raise StateDrift(f"{e.target}: restore could not be verified")
        e.status, e.reverted_at = REVERTED, self._clock()
        return e

    # ---- inspection ----
    def get(self, incident_id) -> Entry:
        if incident_id not in self._entries:
            raise UnknownIncident(incident_id)
        return self._entries[incident_id]

    def __len__(self):
        return len(self._entries)

    def active_ids(self) -> list:
        return [i for i, e in self._entries.items() if e.status == ACTIVE]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(e) for e in self._entries.values()], columns=COLUMNS)

    # ---- persistence ----
    def save(self, path) -> None:
        df = self.to_frame()
        df["verified"] = df["verified"].map(lambda v: "" if v is None else str(bool(v)))
        df.to_csv(path, index=False)

    @classmethod
    def load(cls, path, clock=None) -> "Ledger":
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        led = cls(clock=clock)
        for r in df.to_dict("records"):
            e = Entry(r["incident_id"], r["target"], r["source"], r["action"], r["undo_action"],
                      r["prior_state"], r["applied_state"], float(r["confidence"]), r["approved_by"],
                      r["executed_at"], r["status"], r["reverted_at"],
                      None if r["verified"] == "" else r["verified"] == "True")
            led._entries[e.incident_id] = e
            if e.status == ACTIVE:
                led._state[e.target] = e.applied_state
        return led
