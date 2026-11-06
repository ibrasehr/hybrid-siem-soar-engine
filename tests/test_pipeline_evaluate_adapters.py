import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

from response_engine import (AssetRegistry, Policy, StateDrift, UnknownIncident, load_alerts,
                             load_insider_csv, run_pipeline)
from response_engine.evaluate import sweep

ROOT = Path(__file__).resolve().parents[1]
LOW = AssetRegistry({}, default=1)


# ---------------- pipeline ----------------
def test_autos_are_executed_and_reviews_are_not(mk):
    r = run_pipeline(mk([0.2, 0.7, 0.92, 0.99]), registry=LOW)
    assert len(r.ledger) == 2
    assert r.ledger.get("INC-2").action == "rate_limit_flow"
    assert r.ledger.get("INC-3").action == "block_flow"
    assert r.ledger.get("INC-2").approved_by == "auto"
    assert r.ledger.state_of("flow-1") == "normal"           # review item: nothing executed


def test_every_active_entry_can_be_undone(mk):
    r = run_pipeline(mk([0.92 + i * 0.001 for i in range(60)]), registry=LOW)
    for i in list(r.ledger.active_ids()):
        e = r.ledger.revert(i)
        assert e.verified and r.ledger.state_of(e.target) == "normal"


def test_analyst_approve_executes_proposed_action_and_logs_feedback(mk, clock):
    r = run_pipeline(mk([0.7, 0.8]), registry=LOW, approve_ids=["INC-1"], clock=clock)
    e = r.ledger.get("INC-1")
    assert (e.action, e.approved_by) == ("rate_limit_flow", "analyst")
    fb = r.feedback.to_frame()
    assert list(fb["outcome"]) == ["analyst_approved"] and fb["original_confidence"].iloc[0] == 0.8
    assert len(r.ledger) == 1


def test_analyst_approves_a_bdi_gated_alert_with_the_hard_action(mk):
    r = run_pipeline(mk([0.99]), registry=AssetRegistry({}, default=5), approve_ids=["INC-0"])
    assert r.ledger.get("INC-0").action == "block_flow"


def test_reject_only_writes_feedback(mk):
    r = run_pipeline(mk([0.7]), registry=LOW, reject_ids=["INC-0"])
    assert len(r.ledger) == 0 and list(r.feedback.to_frame()["outcome"]) == ["analyst_rejected"]


def test_revert_flips_state_and_logs_false_positive(mk):
    r = run_pipeline(mk([0.99]), registry=LOW, revert_ids=["INC-0"])
    assert r.ledger.get("INC-0").status == "REVERTED" and r.ledger.state_of("flow-0") == "normal"
    fb = r.feedback.to_frame()
    assert list(fb["outcome"]) == ["reverted_false_positive"] and fb["original_confidence"].iloc[0] == 0.99


def test_approving_a_non_review_or_unknown_alert_is_rejected(mk):
    with pytest.raises(ValueError, match="not in the review queue"):
        run_pipeline(mk([0.99]), registry=LOW, approve_ids=["INC-0"])
    with pytest.raises(ValueError, match="unknown incident"):
        run_pipeline(mk([0.7]), registry=LOW, approve_ids=["nope"])


def test_reverting_something_never_executed_is_rejected(mk):
    with pytest.raises(UnknownIncident):
        run_pipeline(mk([0.7]), registry=LOW, revert_ids=["INC-0"])


def test_pipeline_is_deterministic(mk, clock):
    df = mk([0.3, 0.7, 0.92, 0.99])
    a = run_pipeline(df, registry=LOW, approve_ids=["INC-1"], clock=iter(f"t{i}" for i in range(99)).__next__)
    b = run_pipeline(df, registry=LOW, approve_ids=["INC-1"], clock=iter(f"t{i}" for i in range(99)).__next__)
    assert a.decisions.equals(b.decisions) and a.ledger.to_frame().equals(b.ledger.to_frame())


def test_queue_capacity_comes_from_policy(mk):
    r = run_pipeline(mk([0.6 + i * 0.002 for i in range(40)]), Policy(queue_capacity=10), registry=LOW)
    assert len(r.queue.queue) == 10 and r.queue.overflow_count == 30


# ---------------- evaluate / sweep ----------------
def test_sweep_partitions_every_alert_exactly_once(mk):
    df = mk([0.1, 0.55, 0.7, 0.93, 0.99, 1.0], labels=[0, 0, 1, 1, 0, 1])
    s = sweep(df, registry=LOW)
    assert (s["auto"] + s["review_queue"] + s["log_only"] == s["alerts"]).all()
    assert (s["auto_soft"] + s["auto_hard"] == s["auto"]).all()


def test_sweep_is_monotonic_in_threshold(mk):
    df = mk([i / 100 for i in range(40, 101)])
    s = sweep(df, registry=LOW)
    assert (s["auto"].diff().dropna() <= 0).all()


def test_sweep_hand_computed_example(mk):
    # conf  : .55  .70  .93  .99  1.0       labels: benign, attack, attack, benign, attack
    df = mk([0.55, 0.70, 0.93, 0.99, 1.0], labels=[0, 1, 1, 0, 1])
    row = sweep(df, thresholds=[0.9], registry=LOW).iloc[0]
    assert (row["auto"], row["auto_hard"], row["auto_soft"]) == (3, 2, 1)
    assert row["false_auto_actions"] == 1                         # the 0.99 benign
    assert row["auto_precision"] == pytest.approx(2 / 3)
    assert row["benign_alerts_auto_actioned"] == pytest.approx(1 / 2)
    assert (row["review_queue"], row["attacks_waiting_in_review"]) == (2, 1)


def test_sweep_bdi_gate_moves_alerts_from_auto_to_review(mk):
    df = mk([0.99, 0.99, 0.99], labels=[1, 1, 1])
    crit = AssetRegistry({"flow-0": 5, "flow-1": 1, "flow-2": 1})
    gated = sweep(df, [0.9], registry=crit).iloc[0]
    open_ = sweep(df, [0.9], policy=Policy(bdi_limit=None), registry=crit).iloc[0]
    assert (gated["auto"], gated["bdi_forced_reviews"]) == (2, 1)
    assert (open_["auto"], open_["bdi_forced_reviews"]) == (3, 0)


# ---------------- adapters (uses Ibra's real files when present) ----------------
ALERTS = ROOT / "alerts.json"
needs_repo = pytest.mark.skipif(not ALERTS.exists(), reason="alerts.json not in repo root")


@needs_repo
def test_real_alerts_match_ibras_schema():
    v = Draft7Validator(json.loads((ROOT / "schema.json").read_text()))
    bad = [a["incident_id"] for a in json.loads(ALERTS.read_text()) if list(v.iter_errors(a))]
    assert bad == []


@needs_repo
def test_real_alerts_load_and_label_correctly():
    df = load_alerts(ALERTS)
    assert len(df) == 100 and df["id"].is_unique and set(df["true_label"]) <= {0, 1}
    assert (df.loc[df["family"] == "Normal", "true_label"] == 0).all()
    assert (df.loc[df["family"] != "Normal", "true_label"] == 1).all()
    assert df["target"].str.startswith("flow-").all()


@needs_repo
def test_full_pipeline_runs_on_real_alerts():
    r = run_pipeline(load_alerts(ALERTS))
    assert len(r.decisions) == 100 and (r.decisions["explanation"].str.contains("top drivers")).all()
    assert len(r.ledger) == (r.decisions["tier"] >= 2).sum()


def test_jsonl_and_file_like_inputs(tmp_path):
    a = {"incident_id": "INC-1", "confidence": 0.9, "contributing_event_ids": [1],
         "ground_truth_family": "Normal", "trace": {"rules_fired": [], "top_features": []}}
    b = dict(a, incident_id="INC-2", ground_truth_family="DoS")
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(a) + "\n\n" + json.dumps(b) + "\n")
    df = load_alerts(p)
    assert list(df["true_label"]) == [0, 1] and list(df["id"]) == ["INC-1", "INC-2"]
    import io
    assert len(load_alerts(io.BytesIO(p.read_bytes()))) == 2
    assert len(load_alerts(io.StringIO("[]"))) == 0


def test_insider_loader(tmp_path):
    ok = tmp_path / "i.csv"
    ok.write_text("id,source,confidence,true_label,extra\nu1,insider_threat,0.8,1,zzz\n")
    df = load_insider_csv(ok)
    assert list(df["target"]) == ["u1"] and "extra" not in df.columns
    bad = tmp_path / "b.csv"
    bad.write_text("id,score\nu1,0.5\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_insider_csv(bad)
