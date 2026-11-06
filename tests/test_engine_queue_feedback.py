import json
import math

import pytest

from response_engine import (AssetRegistry, FeedbackLog, Policy, audit_jsonl, audit_records,
                             build_review_queue, decide_all, suggest_threshold)

P = Policy()            # floor .50, auto .90, hard .97, bdi_limit 12, capacity 50
LOW = AssetRegistry({}, default=1)      # every asset low-criticality -> BDI never gates
HIGH = AssetRegistry({}, default=5)     # every asset critical -> hard actions are gated


# ---------------- tiers & boundaries ----------------
@pytest.mark.parametrize("conf,tier", [
    (0.0, 0), (0.4999, 0), (0.50, 1), (0.8999, 1), (0.90, 2), (0.9699, 2), (0.97, 3), (1.0, 3)])
def test_tier_boundaries_are_inclusive_at_the_lower_edge(mk, conf, tier):
    assert decide_all(mk([conf]), policy=P, registry=LOW)["tier"].iloc[0] == tier


def test_decision_labels_and_actions_per_tier(mk):
    d = decide_all(mk([0.2, 0.7, 0.92, 0.99]), policy=P, registry=LOW)
    assert list(d["decision"]) == ["LOG ONLY", "HUMAN REVIEW", "AUTO-RESPONSE", "AUTO-RESPONSE"]
    assert list(d["action"]) == ["log_only", "flag_for_analyst", "rate_limit_flow", "block_flow"]
    assert list(d["undo_action"]) == ["", "", "restore_flow_rate", "unblock_flow"]
    assert list(d["tier_name"]) == ["LOG_ONLY", "REVIEW", "AUTO_SOFT", "AUTO_HARD"]


def test_insider_source_uses_its_own_catalog(mk):
    d = decide_all(mk([0.92, 0.99], source="insider_threat"), policy=P, registry=LOW)
    assert list(d["action"]) == ["step_up_mfa", "suspend_session"]


def test_legacy_threshold_argument_still_works(mk):
    d = decide_all(mk([0.80, 0.95]), 0.75, registry=LOW)
    assert list(d["decision"]) == ["AUTO-RESPONSE", "AUTO-RESPONSE"]
    low = decide_all(mk([0.2, 0.4]), 0.3, registry=LOW)        # threshold below the default floor
    assert list(low["tier"]) == [0, 2]                         # floor follows it down: 0.2 logged, 0.4 acts


def test_decisions_never_depend_on_ground_truth(mk):
    a = mk([0.3, 0.8, 0.95, 0.99], labels=[1, 1, 0, 0])
    b = mk([0.3, 0.8, 0.95, 0.99], labels=[0, 0, 1, 1])
    b["family"] = "Exploits"
    da, db = decide_all(a, policy=P), decide_all(b, policy=P)
    for col in ("tier", "action", "explanation", "bdi"):
        assert list(da[col]) == list(db[col])


# ---------------- BDI gate ----------------
def test_bdi_gate_forces_review_and_proposes_the_hard_action(mk):
    d = decide_all(mk([0.99]), policy=P, registry=HIGH)        # 5 x 5 = 25 > 12
    r = d.iloc[0]
    assert (r["tier"], r["review_reason"], r["proposed_action"]) == (1, "bdi_gate", "block_flow")
    assert r["bdi"] == 25 and r["action"] == "flag_for_analyst"
    assert "BDI 25" in r["explanation"] and "> limit 12" in r["explanation"]


def test_soft_action_is_never_gated_at_default_limit(mk):
    r = decide_all(mk([0.92]), policy=P, registry=HIGH).iloc[0]      # 5 x 2 = 10 <= 12
    assert r["tier"] == 2


def test_gate_disabled_with_none(mk):
    r = decide_all(mk([0.99]), policy=Policy(bdi_limit=None), registry=HIGH).iloc[0]
    assert r["tier"] == 3


def test_bdi_exactly_at_limit_is_allowed(mk):
    reg = AssetRegistry({"flow-0": 3})                          # hard: 3 x 5 = 15
    assert decide_all(mk([0.99]), policy=Policy(bdi_limit=15), registry=reg)["tier"].iloc[0] == 3
    assert decide_all(mk([0.99]), policy=Policy(bdi_limit=14.9), registry=reg)["tier"].iloc[0] == 1


def test_low_confidence_review_proposes_the_soft_action(mk):
    r = decide_all(mk([0.7]), policy=P, registry=LOW).iloc[0]
    assert (r["review_reason"], r["proposed_action"]) == ("below_auto_threshold", "rate_limit_flow")


# ---------------- explanations & audit ----------------
def test_explanation_includes_rules_and_top_drivers(mk):
    df = mk([0.99])
    df.at[0, "rules_fired"] = ["rule_a", "rule_b"]
    df.at[0, "top_features"] = [{"feature": "sbytes", "z_score": 12.34}, {"feature": "ct_srv_dst"}]
    e = decide_all(df, policy=P, registry=LOW)["explanation"].iloc[0]
    assert "rule_a, rule_b" in e and "sbytes (z=12.3)" in e and "ct_srv_dst" in e
    assert "hard threshold 0.97" in e and "AUTO_HARD block_flow" in e


def test_explanation_for_missing_trace(mk):
    e = decide_all(mk([0.99]), policy=P, registry=LOW)["explanation"].iloc[0]
    assert "rules fired: none" in e and "no feature trace supplied" in e


def test_audit_records_have_policy_and_no_ground_truth(mk):
    d = decide_all(mk([0.6, 0.99], labels=[1, 0]), policy=P, registry=LOW)
    recs = list(audit_records(d, P, LOW))
    assert len(recs) == 2
    for r in recs:
        assert "true_label" not in r and "family" not in r
        assert r["policy"] == P.to_dict() and r["asset_criticality_simulated"] is False
    parsed = [json.loads(line) for line in audit_jsonl(d, P, LOW).splitlines()]
    assert parsed[1]["action"] == "block_flow"
    assert list(audit_records(d, P))[0]["asset_criticality_simulated"] is True


# ---------------- fail closed / robustness ----------------
def test_unknown_source_fails_closed(mk):
    from response_engine import UnknownAction
    with pytest.raises(UnknownAction):
        decide_all(mk([0.99], source="mystery"), policy=P)


@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.5, None])
def test_invalid_confidence_never_auto_acts(mk, bad):
    with pytest.raises(ValueError, match="confidence"):
        decide_all(mk([0.99, bad]), policy=P)


def test_empty_input_returns_expected_columns(mk):
    d = decide_all(mk([]), policy=P)
    assert len(d) == 0 and {"tier", "decision", "bdi", "explanation"} <= set(d.columns)


def test_decide_all_is_deterministic_and_does_not_mutate_input(mk):
    df = mk([0.5, 0.95, 0.99])
    before = df.copy(deep=True)
    a, b = decide_all(df, policy=P), decide_all(df, policy=P)
    assert a.equals(b) and df.equals(before) and "tier" not in df.columns


# ---------------- review queue ----------------
def test_queue_respects_capacity_and_never_drops(mk):
    d = decide_all(mk([0.5 + i * 0.004 for i in range(100)]), policy=P, registry=LOW)  # all < .90
    q = build_review_queue(d, capacity=30)
    assert len(q.queue) == 30 and q.overflow_count == (d["tier"] == 1).sum() - 30
    assert len(q.queue) + q.overflow_count == (d["tier"] == 1).sum()
    assert set(q.overflow["overflow_reason"]) == {"queue_capacity"}
    assert not set(q.queue["id"]) & set(q.overflow["id"])


def test_queue_orders_by_confidence_desc_then_id(mk):
    d = decide_all(mk([0.6, 0.8, 0.8, 0.7]), policy=P, registry=LOW)
    q = build_review_queue(d, capacity=10)
    assert list(q.queue["confidence"]) == [0.8, 0.8, 0.7, 0.6]
    assert list(q.queue["id"])[:2] == ["INC-1", "INC-2"]


def test_queue_excludes_auto_and_log_only_and_handles_empty(mk):
    d = decide_all(mk([0.2, 0.95, 0.99]), policy=P, registry=LOW)
    q = build_review_queue(d, 5)
    assert len(q.queue) == 0 and q.overflow_count == 0


@pytest.mark.parametrize("cap", [0, -1, 2.5, True])
def test_queue_rejects_bad_capacity(mk, cap):
    with pytest.raises(ValueError):
        build_review_queue(decide_all(mk([0.7]), policy=P), cap)


# ---------------- feedback ----------------
def test_feedback_validates_and_records(clock):
    fb = FeedbackLog(clock=clock)
    fb.record("A", "correlation", 0.82, "HUMAN REVIEW", "analyst_approved")
    assert len(fb) == 1 and fb.to_frame()["timestamp"].iloc[0] == "t0001"
    with pytest.raises(ValueError):
        fb.record("B", "correlation", 0.8, "x", "maybe")
    with pytest.raises(ValueError):
        fb.record("B", "correlation", 1.8, "x", "analyst_approved")
    assert len(fb) == 1


def test_feedback_persists_and_appends_across_instances(tmp_path):
    p = str(tmp_path / "feedback_log.csv")
    FeedbackLog(p).record("A", "correlation", 0.8, "HUMAN REVIEW", "analyst_approved")
    FeedbackLog(p).record("B", "correlation", 0.9, "AUTO-RESPONSE", "reverted_false_positive")
    import pandas as pd
    df = pd.read_csv(p)
    assert list(df["incident_id"]) == ["A", "B"] and len(df.columns) == 7


def _fb(rows):
    fb = FeedbackLog()
    for i, (c, outcome) in enumerate(rows):
        fb.record(f"I{i}", "correlation", c, "x", outcome)
    return fb.to_frame()


def test_suggest_threshold_needs_data():
    assert suggest_threshold(_fb([]))["suggested_threshold"] is None
    r = suggest_threshold(_fb([(0.9, "analyst_approved")] * 5), min_n=20)
    assert r["suggested_threshold"] is None and "n >= 20" in r["reason"]


def test_suggest_threshold_picks_lowest_threshold_meeting_precision():
    rows = ([(0.6, "analyst_rejected")] * 10 + [(0.7, "reverted_false_positive")] * 10
            + [(0.95, "analyst_approved")] * 30)
    r = suggest_threshold(_fb(rows), target_precision=0.95, min_n=20)
    assert r["suggested_threshold"] == 0.95 and r["support"] == 30
    assert r["applied"] is False                                  # never auto-applied


def test_suggest_threshold_returns_none_when_target_unreachable():
    rows = [(0.9, "analyst_approved")] * 20 + [(0.9, "analyst_rejected")] * 20
    assert suggest_threshold(_fb(rows), 0.95, 20)["suggested_threshold"] is None
