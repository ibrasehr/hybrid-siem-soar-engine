"""Fixed action catalog. The engine can only ever run actions listed here.

This is a deliberate design choice (no open-ended / LLM-generated actions):
anything not in CATALOG fails closed with UnknownAction.
"""
from __future__ import annotations

from dataclasses import dataclass

SEVERITY = {"soft": 2, "hard": 5}          # used by the Business Disruption Index


class UnknownAction(ValueError):
    """Requested action/source/kind is not in the approved catalog."""


@dataclass(frozen=True)
class Action:
    name: str        # what we execute
    undo: str        # its exact inverse
    state: str       # state the target is in while the action is active
    kind: str        # "soft" or "hard"

    @property
    def severity(self) -> int:
        return SEVERITY[self.kind]


CATALOG = {
    "correlation": {
        "soft": Action("rate_limit_flow", "restore_flow_rate", "flow_rate_limited", "soft"),
        "hard": Action("block_flow", "unblock_flow", "flow_blocked", "hard"),
    },
    "insider_threat": {
        "soft": Action("step_up_mfa", "clear_mfa_challenge", "mfa_challenged", "soft"),
        "hard": Action("suspend_session", "restore_session", "session_suspended", "hard"),
    },
}


def action_for(source: str, kind: str) -> Action:
    try:
        return CATALOG[source][kind]
    except KeyError:
        raise UnknownAction(f"no catalog action for source={source!r}, kind={kind!r}") from None


def find_action(source: str, name: str) -> Action:
    for act in CATALOG.get(source, {}).values():
        if act.name == name:
            return act
    raise UnknownAction(f"{name!r} is not an approved action for source {source!r}")
