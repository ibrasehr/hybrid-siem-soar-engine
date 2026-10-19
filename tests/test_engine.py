import json
import os
import sys
import jsonschema
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from correlation_engine import CorrelationEngine, LEAKAGE_COLUMNS


@pytest.fixture
def sample_data():
    """Create a synthetic dataset matching the UNSW-NB15 schema."""
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "UNSW_NB15_training-set.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, nrows=500)

    # Fallback synthetic dataframe
    n = 200
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "id": np.arange(n),
        "dur": rng.exponential(0.1, n),
        "spkts": rng.poisson(10, n),
        "dpkts": rng.poisson(8, n),
        "sbytes": rng.poisson(500, n),
        "dbytes": rng.poisson(800, n),
        "rate": rng.uniform(10, 50000, n),
        "sttl": rng.choice([31, 62, 254], n),
        "dttl": rng.choice([29, 60, 252], n),
        "sload": rng.uniform(100, 100000, n),
        "dload": rng.uniform(100, 100000, n),
        "ct_srv_src": rng.randint(1, 40, n),
        "ct_state_ttl": rng.randint(0, 6, n),
        "ct_dst_ltm": rng.randint(1, 40, n),
        "ct_dst_src_ltm": rng.randint(1, 40, n),
        "ct_dst_sport_ltm": rng.randint(1, 40, n),
        "ct_src_dport_ltm": rng.randint(1, 40, n),
        "attack_cat": ["Normal" if i % 2 == 0 else "Exploits" for i in range(n)],
        "label": [0 if i % 2 == 0 else 1 for i in range(n)],
    })


def test_no_leakage_columns_in_feature_list(sample_data):
    """Verify that no target or leakage columns are present in features."""
    engine = CorrelationEngine(n_estimators=10)
    engine.fit(sample_data, label_col="attack_cat", benign="Normal")

    forbidden = {col.lower() for col in LEAKAGE_COLUMNS}
    for feat in engine.features:
        assert feat.lower() not in forbidden, f"Leakage column found in features: {feat}"
    assert "label" not in [f.lower() for f in engine.features]
    assert "id" not in [f.lower() for f in engine.features]


def test_alerts_validate_against_schema(sample_data):
    """Verify that generated alerts strictly conform to schema.json."""
    engine = CorrelationEngine(n_estimators=10)
    engine.fit(sample_data, label_col="attack_cat", benign="Normal")

    alerts = engine.predict(sample_data, threshold=0.1)
    assert len(alerts) > 0, "Expected at least one alert at low threshold"

    schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    for alert in alerts:
        jsonschema.validate(instance=alert, schema=schema)
        assert "rules_fired" in alert["trace"]
        assert isinstance(alert["trace"]["rules_fired"], list)
        assert len(alert["trace"]["top_features"]) > 0
        for feat in alert["trace"]["top_features"]:
            assert "feature" in feat
            assert "importance" in feat
            assert "observed_value" in feat


def test_predict_empty_on_threshold_above_one(sample_data):
    """Verify that predict() returns an empty list when threshold is 1.01."""
    engine = CorrelationEngine(n_estimators=10)
    engine.fit(sample_data, label_col="attack_cat", benign="Normal")

    alerts = engine.predict(sample_data, threshold=1.01)
    assert alerts == [], f"Expected empty list for threshold 1.01, got {len(alerts)} alerts"
