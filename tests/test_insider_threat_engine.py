
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
