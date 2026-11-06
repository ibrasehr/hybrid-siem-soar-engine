"""Decision engine (Harshith). Pure functions, no Streamlit.

Tiers:  0 LOG_ONLY   confidence < review_floor
        1 REVIEW     below auto_threshold, OR a BDI-gated action (analyst must approve)
        2 AUTO_SOFT  auto_threshold <= confidence < hard_threshold
        3 AUTO_HARD  confidence >= hard_threshold
Decisions read ONLY confidence, source and target (the trace only feeds the text).
true_label / family are never read here, so ground truth cannot leak in.
"""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from .assets import AssetRegistry
from .catalog import action_for
from .policy import (DECISION_LABELS, TIER_AUTO_HARD, TIER_AUTO_SOFT, TIER_LOG, TIER_NAMES,
                     TIER_REVIEW, Policy)

OUT_COLUMNS = ["tier", "tier_name", "decision", "action", "undo_action", "proposed_action",
               "review_reason", "asset_criticality", "bdi"]


def _as_list(x):
    return x if isinstance(x, list) else []


def _drivers(top_features, n=3) -> str:
    parts = []
    for f in _as_list(top_features)[:n]:
        z = f.get("z_score")
        parts.append(f"{f['feature']} (z={z:.1f})" if z is not None else f["feature"])
    return ", ".join(parts)


def explain(source, confidence, tier, policy, reason, action, proposed, crit, bdi_value,
            rules, top_features) -> str:
    c = f"{source} confidence {confidence:.2f}"
    if tier == TIER_LOG:
        msg = f"{c} < review floor {policy.review_floor:.2f} -> LOG ONLY"
    elif tier == TIER_REVIEW and reason == "bdi_gate":
        msg = (f"{c} >= auto threshold {policy.auto_threshold:.2f} but BDI {bdi_value} "
               f"(criticality {crit} x severity {bdi_value // crit}) > limit {policy.bdi_limit:g} "
               f"-> HUMAN REVIEW (proposed: {proposed})")
    elif tier == TIER_REVIEW:
        msg = (f"{c} < auto threshold {policy.auto_threshold:.2f} -> HUMAN REVIEW "
               f"(proposed: {proposed})")
    elif tier == TIER_AUTO_SOFT:
        msg = (f"{c} >= auto threshold {policy.auto_threshold:.2f} and < hard threshold "
               f"{policy.hard_threshold:.2f} -> AUTO_SOFT {action} (BDI {bdi_value})")
    else:
        msg = (f"{c} >= hard threshold {policy.hard_threshold:.2f} -> AUTO_HARD {action} "
               f"(BDI {bdi_value})")
    if source == "correlation":
        r = _as_list(rules)
        msg += f"; rules fired: {', '.join(r) if r else 'none'}"
    d = _drivers(top_features)
    msg += f"; top drivers: {d}" if d else "; no feature trace supplied"
    return msg


def decide_all(df: pd.DataFrame, threshold: Optional[float] = None, policy: Optional[Policy] = None,
               registry: Optional[AssetRegistry] = None, with_explanations: bool = True) -> pd.DataFrame:
    """Apply the policy to every alert. `threshold` (legacy arg) overrides auto_threshold."""
    policy = policy or Policy()
    if threshold is not None:
        policy = policy.with_auto_threshold(threshold)
    registry = registry or AssetRegistry()

    conf = pd.to_numeric(df["confidence"], errors="coerce")
    if conf.isna().any() or ((conf < 0) | (conf > 1)).any():       # fail closed: NaN must never auto-act
        raise ValueError("confidence must be a number in [0, 1] for every alert")

    tiers, actions, undos, proposed, reasons, crits, bdis = [], [], [], [], [], [], []
    crit_cache: dict = {}
    for s, t, c in zip(df["source"], df["target"], df["confidence"]):
        if c < policy.review_floor:
            tiers.append(TIER_LOG); actions.append("log_only"); undos.append("")
            proposed.append(""); reasons.append("below_review_floor"); crits.append(0); bdis.append(0)
            continue
        kind = "hard" if c >= policy.hard_threshold else "soft"
        a = action_for(s, kind)                                   # fails closed on unknown source
        if t not in crit_cache:
            crit_cache[t] = registry.criticality(t)
        crit = crit_cache[t]
        b = crit * a.severity
        crits.append(crit); bdis.append(b); proposed.append(a.name)
        if c < policy.auto_threshold:
            tiers.append(TIER_REVIEW); actions.append("flag_for_analyst"); undos.append("")
            reasons.append("below_auto_threshold")
        elif policy.bdi_limit is not None and b > policy.bdi_limit:
            tiers.append(TIER_REVIEW); actions.append("flag_for_analyst"); undos.append("")
            reasons.append("bdi_gate")
        else:
            tiers.append(TIER_AUTO_HARD if kind == "hard" else TIER_AUTO_SOFT)
            actions.append(a.name); undos.append(a.undo); reasons.append("")

    out = df.copy()
    out["tier"] = tiers
    out["tier_name"] = [TIER_NAMES[t] for t in tiers]
    out["decision"] = [DECISION_LABELS[t] for t in tiers]
    out["action"] = actions
    out["undo_action"] = undos
    out["proposed_action"] = proposed
    out["review_reason"] = reasons
    out["asset_criticality"] = crits
    out["bdi"] = bdis
    if with_explanations:
        out["explanation"] = [
            explain(s, c, t, policy, rr, a, p, cr, b, r, f)
            for s, c, t, rr, a, p, cr, b, r, f in zip(
                out["source"], out["confidence"], tiers, reasons, actions, proposed,
                crits, bdis, out["rules_fired"], out["top_features"])]
    return out


def audit_records(decisions: pd.DataFrame, policy: Optional[Policy] = None,
                  registry: Optional[AssetRegistry] = None):
    """One structured, auditable record per decision. Never includes ground truth."""
    policy = policy or Policy()
    simulated = (registry or AssetRegistry()).simulated
    for r in decisions.to_dict("records"):
        yield {
            "incident_id": r["id"], "source": r["source"], "target": r["target"],
            "confidence": r["confidence"], "tier": int(r["tier"]), "tier_name": r["tier_name"],
            "decision": r["decision"], "action": r["action"], "undo_action": r["undo_action"],
            "proposed_action": r["proposed_action"], "review_reason": r["review_reason"],
            "asset_criticality": int(r["asset_criticality"]),
            "asset_criticality_simulated": simulated, "bdi": int(r["bdi"]),
            "rules_fired": _as_list(r["rules_fired"]), "top_features": _as_list(r["top_features"]),
            "policy": policy.to_dict(),
        }


def audit_jsonl(decisions, policy=None, registry=None) -> str:
    return "\n".join(json.dumps(r) for r in audit_records(decisions, policy, registry))
