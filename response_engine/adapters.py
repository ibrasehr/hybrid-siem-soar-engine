"""Adapters: turn upstream module outputs into one flat table the response engine reads.

Correlation (Ibra)  -> alerts.json / alerts_full.jsonl, validated by schema.json
Insider threat (Divya) -> CSV with columns: id, source, confidence, true_label
"""
import io
import json
from pathlib import Path

import pandas as pd

REQUIRED_INSIDER_COLS = ["id", "source", "confidence", "true_label"]
OUT_COLS = ["id", "target", "source", "confidence", "true_label", "family",
            "rules_fired", "top_features"]


def _read_text(source):
    """Accept a path or an uploaded file-like object."""
    if hasattr(source, "read"):
        raw = source.read()
        return raw.decode("utf-8") if isinstance(raw, bytes) else raw
    return Path(source).read_text(encoding="utf-8")


def _parse_alerts(text):
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):                       # alerts.json = JSON array
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]  # .jsonl


def load_alerts(source):
    """Load Ibra's correlation alerts into the flat table.

    true_label comes from ground_truth_family, which the schema marks
    evaluation_only. It is kept ONLY so we can score our own decisions;
    engine.decide_all never reads it.
    """
    alerts = _parse_alerts(_read_text(source))
    rows = []
    for a in alerts:
        family = str(a["ground_truth_family"]).strip()
        event_id = a["contributing_event_ids"][0]
        rows.append({
            "id": a["incident_id"],
            # UNSW-NB15 rows are network flows with no host/IP, so the action
            # target is the flow itself. Real deployment would map flow -> host.
            "target": f"flow-{event_id}",
            "source": "correlation",
            "confidence": float(a["confidence"]),
            "true_label": int(family.lower() != "normal"),
            "family": family,
            "rules_fired": list(a["trace"].get("rules_fired", [])),
            "top_features": list(a["trace"].get("top_features", [])),
        })
    return pd.DataFrame(rows, columns=OUT_COLS)


def load_insider_csv(source):
    """Load Divya's insider-threat predictions (no feature trace yet)."""
    df = pd.read_csv(io.StringIO(_read_text(source)))
    missing = [c for c in REQUIRED_INSIDER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"insider CSV is missing columns {missing}; "
                         f"expected {REQUIRED_INSIDER_COLS}")
    df = df[REQUIRED_INSIDER_COLS].copy()
    df["target"] = df["id"]
    df["family"] = ""
    df["rules_fired"] = [[] for _ in range(len(df))]
    df["top_features"] = [[] for _ in range(len(df))]
    return df[OUT_COLS]
