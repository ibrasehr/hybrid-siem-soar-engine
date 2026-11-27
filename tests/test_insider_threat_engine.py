
from insider_threat_engine import InsiderThreatEngine
import pandas as pd


def test_engine_loads():
    engine = InsiderThreatEngine()
    assert engine.model is not None


def test_prediction_output():
    engine = InsiderThreatEngine()

    sample = pd.DataFrame([{
        "event_id": "E001",
        "timestamp": "2025-01-01 10:00:00",
        "user": "ABC1234",
        "pc": "PC-01",
        "activity": "Logon",
        "event_type": "logon"
    }])

    result = engine.predict(sample)

    assert len(result) == 1
    assert "predicted_label" in result.columns
    assert "confidence" in result.columns
    assert "rule_alert" in result.columns


def test_confidence_range():
    engine = InsiderThreatEngine()

    sample = pd.DataFrame([{
        "event_id": "E002",
        "timestamp": "2025-01-01 11:00:00",
        "user": "XYZ1234",
        "pc": "PC-02",
        "activity": "Logon",
        "event_type": "logon"
    }])

    result = engine.predict(sample)

    assert result["confidence"].between(0, 1).all()
    assert (result["source"] == "insider_threat").all()


def test_empty_events_returns_empty_dataframe():
    engine = InsiderThreatEngine()
    empty = pd.DataFrame(columns=["event_id", "timestamp", "user", "pc", "activity", "event_type"])

    result = engine.predict(empty)

    assert result.empty
    assert list(result.columns) == [
        "user",
        "source",
        "confidence",
        "predicted_label",
        "rule_alert",
        "rule_reasons",
    ]


def test_apply_rules_flags_anomalous_activity():
    engine = InsiderThreatEngine()
    features = pd.DataFrame([{
        "user": "BAD0001",
        "total_events": 120,
        "http_count": 105,
        "device_count": 6,
        "email_count": 1,
        "logon_count": 2,
        "file_count": 4,
        "after_hours_count": 8,
        "weekend_count": 1,
        "active_days": 2,
    }])

    rules = engine.apply_rules(features)

    assert rules["rule_alert"].iloc[0] == 1
    assert set(rules["rule_reasons"].iloc[0]) == {
        "High event volume",
        "Excessive web activity",
        "Unusual device usage",
        "Excessive file activity",
        "Unusual after-hours activity",
    }


def test_soar_pipeline_integration():
    from pathlib import Path
    from response_engine import load_insider_csv, run_pipeline

    csv_path = Path(__file__).resolve().parents[1] / "data" / "insider_threat_predictions.csv"
    df = load_insider_csv(csv_path)

    result = run_pipeline(df)

    assert len(result.decisions) == len(df)
    assert (result.decisions["source"] == "insider_threat").all()
