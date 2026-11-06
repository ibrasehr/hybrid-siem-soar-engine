import pytest

from response_engine import (CATALOG, AssetRegistry, Policy, UnknownAction, action_for, bdi,
                             find_action)


# ---------------- Policy ----------------
def test_policy_defaults_are_valid():
    p = Policy()
    assert p.review_floor <= p.auto_threshold <= p.hard_threshold


@pytest.mark.parametrize("kw", [
    dict(review_floor=0.95, auto_threshold=0.9),
    dict(auto_threshold=0.99, hard_threshold=0.9),
    dict(auto_threshold=1.2), dict(review_floor=-0.1), dict(auto_threshold=float("nan")),
    dict(auto_threshold=True), dict(bdi_limit=0), dict(bdi_limit=-3),
    dict(queue_capacity=0), dict(queue_capacity=2.5), dict(queue_capacity=True),
])
def test_policy_rejects_invalid(kw):
    with pytest.raises(ValueError):
        Policy(**kw)


def test_policy_allows_disabled_gate():
    assert Policy(bdi_limit=None).bdi_limit is None


def test_policy_is_immutable():
    with pytest.raises(Exception):
        Policy().auto_threshold = 0.1


def test_with_auto_threshold_keeps_ordering():
    p = Policy().with_auto_threshold(0.99)
    assert p.auto_threshold == 0.99 and p.hard_threshold >= 0.99
    q = Policy().with_auto_threshold(0.3)
    assert q.review_floor <= 0.3
    with pytest.raises(ValueError):
        Policy().with_auto_threshold(1.5)


def test_policy_from_dict_roundtrip_and_unknown_key():
    p = Policy(auto_threshold=0.8, queue_capacity=10)
    assert Policy.from_dict(p.to_dict()) == p
    with pytest.raises(ValueError, match="unknown policy keys"):
        Policy.from_dict({"nope": 1})


# ---------------- Assets / BDI ----------------
def test_simulated_criticality_is_deterministic_and_in_range():
    a, b = AssetRegistry(), AssetRegistry()
    vals = [a.criticality(f"flow-{i}") for i in range(500)]
    assert vals == [b.criticality(f"flow-{i}") for i in range(500)]
    assert set(vals) <= {1, 2, 3, 4, 5} and len(set(vals)) == 5
    assert a.simulated is True


def test_real_registry_mapping_default_and_flag():
    r = AssetRegistry({"db-1": 5}, default=2)
    assert (r.criticality("db-1"), r.criticality("unknown")) == (5, 2)
    assert r.simulated is False


def test_registry_rejects_bad_criticality(tmp_path):
    with pytest.raises(ValueError):
        AssetRegistry({"x": 9})
    with pytest.raises(ValueError):
        AssetRegistry({"x": 2.5})
    bad = tmp_path / "a.csv"
    bad.write_text("host,crit\na,1\n")
    with pytest.raises(ValueError, match="needs columns"):
        AssetRegistry.from_csv(bad)
    good = tmp_path / "b.csv"
    good.write_text("target,criticality\nflow-1,4\n")
    assert AssetRegistry.from_csv(good).criticality("flow-1") == 4


def test_bdi_is_criticality_times_severity():
    assert bdi(3, "soft") == 6 and bdi(3, "hard") == 15 and bdi(1, "hard") == 5


# ---------------- Catalog ----------------
def test_every_catalog_action_has_distinct_inverse_and_state():
    for source, kinds in CATALOG.items():
        for kind, act in kinds.items():
            assert act.kind == kind and act.name != act.undo
            assert action_for(source, kind) is act
            assert find_action(source, act.name) is act
    names = [a.name for k in CATALOG.values() for a in k.values()]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("source,kind", [("nope", "soft"), ("correlation", "nuke"), ("", "")])
def test_unknown_source_or_kind_fails_closed(source, kind):
    with pytest.raises(UnknownAction):
        action_for(source, kind)


def test_unapproved_action_name_is_rejected():
    with pytest.raises(UnknownAction):
        find_action("correlation", "rm_rf_everything")
    with pytest.raises(UnknownAction):          # valid name, wrong source
        find_action("correlation", "suspend_session")
