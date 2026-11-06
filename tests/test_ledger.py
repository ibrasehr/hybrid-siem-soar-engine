import pytest

from response_engine import (AlreadyContained, AlreadyReverted, Ledger, StateDrift, UnknownAction,
                             UnknownIncident)


def run(led, i="A", target="flow-1", action="block_flow", by="auto", c=0.99):
    return led.execute(i, target, "correlation", action, c, by)


def test_execute_changes_state_and_snapshots_prior():
    led = Ledger()
    e = run(led)
    assert led.state_of("flow-1") == "flow_blocked"
    assert (e.prior_state, e.applied_state, e.undo_action) == ("normal", "flow_blocked", "unblock_flow")
    assert e.status == "ACTIVE" and e.verified is None


def test_revert_restores_state_and_verifies():
    led = Ledger()
    run(led)
    e = led.revert("A")
    assert led.state_of("flow-1") == "normal"
    assert (e.status, e.verified) == ("REVERTED", True) and e.reverted_at


def test_reverting_one_incident_leaves_others_untouched():
    led = Ledger()
    run(led, "A", "flow-1"); run(led, "B", "flow-2")
    led.revert("A")
    assert led.state_of("flow-2") == "flow_blocked" and led.get("B").status == "ACTIVE"


def test_execute_is_idempotent_for_same_incident():
    led = Ledger()
    first, second = run(led), run(led)
    assert first is second and len(led) == 1


def test_second_incident_on_contained_target_is_refused():
    led = Ledger()
    run(led, "A")
    with pytest.raises(AlreadyContained):
        run(led, "B", action="rate_limit_flow")
    assert len(led) == 1


def test_revert_twice_and_reexecute_after_revert_are_refused():
    led = Ledger()
    run(led); led.revert("A")
    with pytest.raises(AlreadyReverted):
        led.revert("A")
    with pytest.raises(AlreadyReverted):
        run(led)


def test_revert_unknown_incident():
    with pytest.raises(UnknownIncident):
        Ledger().revert("nope")
    with pytest.raises(UnknownIncident):
        Ledger().get("nope")


def test_unapproved_action_is_refused_and_nothing_is_recorded():
    led = Ledger()
    with pytest.raises(UnknownAction):
        run(led, action="format_disk")
    assert len(led) == 0 and led.state_of("flow-1") == "normal"


def test_state_drift_blocks_revert_instead_of_guessing():
    led = Ledger()
    run(led)
    led._state["flow-1"] = "something_else"            # someone changed it behind our back
    with pytest.raises(StateDrift):
        led.revert("A")
    assert led.get("A").status == "ACTIVE"             # not silently marked reverted


def test_clock_injection_is_deterministic(clock):
    led = Ledger(clock=clock)
    run(led); led.revert("A")
    e = led.get("A")
    assert (e.executed_at, e.reverted_at) == ("t0001", "t0002")


def test_save_load_roundtrip_restores_state(tmp_path):
    led = Ledger()
    run(led, "A", "flow-1"); run(led, "B", "flow-2"); led.revert("B")
    p = tmp_path / "ledger.csv"
    led.save(p)
    back = Ledger.load(p)
    assert back.state_of("flow-1") == "flow_blocked" and back.state_of("flow-2") == "normal"
    assert back.get("B").status == "REVERTED" and back.get("B").verified is True
    assert back.get("A").verified is None
    assert back.to_frame().equals(led.to_frame())
    back.revert("A")                                    # loaded ledger is still operable
    assert back.state_of("flow-1") == "normal"


def test_active_ids_and_frame_columns():
    led = Ledger()
    run(led, "A", "flow-1"); run(led, "B", "flow-2"); led.revert("A")
    assert led.active_ids() == ["B"]
    assert {"incident_id", "undo_action", "prior_state", "status", "verified"} <= set(led.to_frame().columns)
