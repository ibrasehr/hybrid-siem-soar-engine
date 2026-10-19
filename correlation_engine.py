"""
CorrelationEngine: production-ready SIEM correlation engine prototype.
Combines Random Forest classification with hand-crafted behavioral rule tracing
and schema-validated incident alert generation.
"""

import argparse
import json
import os
import joblib
import jsonschema
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

LEAKAGE_COLUMNS = {"label", "id", "attack_cat", "family", "is_attack", "event_id"}


def rule_high_packet_rate(df_or_row, limits):
    """Targets volumetric DoS and flooding attacks by detecting packet transmission rates exceeding benign norms."""
    return df_or_row["rate"] > limits["rate"]


def rule_packet_asymmetry(df_or_row, limits):
    """Targets scanning, reconnaissance, and unilateral floods where outbound source packets vastly exceed inbound responses."""
    ratio = df_or_row["spkts"] / (df_or_row["dpkts"] + 1)
    return ratio > limits["spkts_ratio"]


def rule_high_srv_connections(df_or_row, limits):
    """Targets service-level port scans and brute-force sweeps that create dense connection counts to the same service."""
    return df_or_row["ct_srv_src"] > limits["ct_srv_src"]


def rule_high_dst_density(df_or_row, limits):
    """Targets host scanning, probing, and DoS targeting a concentrated destination within a short time window."""
    return df_or_row["ct_dst_ltm"] > limits["ct_dst_ltm"]


def rule_short_duration_burst(df_or_row, limits):
    """Targets exploit attempts and fast probe bursts delivering high packet volumes over near-zero connection duration."""
    burst = df_or_row["spkts"] / (df_or_row["dur"] + 0.001)
    return burst > limits["burst_density"]


RULE_MAP = {
    "rule_high_packet_rate": rule_high_packet_rate,
    "rule_packet_asymmetry": rule_packet_asymmetry,
    "rule_high_srv_connections": rule_high_srv_connections,
    "rule_high_dst_density": rule_high_dst_density,
    "rule_short_duration_burst": rule_short_duration_burst,
}


def load_schema():
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


class CorrelationEngine:
    """Hybrid SIEM Correlation Engine combining a supervised Random Forest
    with hand-crafted behavioral rule limits derived from benign training traffic."""

    def __init__(self, n_estimators=100, pct=99, random_state=42):
        self.n_estimators = n_estimators
        self.pct = pct
        self.random_state = random_state
        self.rf = None
        self.features = []
        self.limits = {}
        self.feature_importances_ = {}
        self.top_feature_names_ = []
        self._schema = load_schema()

    def _extract_features(self, df, label_col=None):
        exclude = {col.lower() for col in LEAKAGE_COLUMNS}
        if label_col:
            exclude.add(label_col.lower())
        num = df.select_dtypes(include=[np.number]).columns
        return [c for c in num if c.lower() not in exclude]

    def fit(self, df, label_col="attack_cat", benign="Normal"):
        """Fit Random Forest and compute rule baseline limits on benign training traffic."""
        clean_df = df.copy()
        clean_df.columns = [c.strip() for c in clean_df.columns]
        clean_df = clean_df.replace([np.inf, -np.inf], np.nan).dropna()

        # Target column handling
        clean_df["family"] = clean_df[label_col].astype(str).str.strip()
        clean_df["is_attack"] = (clean_df["family"].str.lower() != benign.lower()).astype(int)

        self.features = self._extract_features(clean_df, label_col=label_col)

        # Train Random Forest
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.rf.fit(clean_df[self.features], clean_df["is_attack"])

        # Record feature importances
        imp_series = pd.Series(self.rf.feature_importances_, index=self.features)
        self.feature_importances_ = imp_series.to_dict()
        self.top_feature_names_ = list(imp_series.nlargest(5).index)

        # Fit hand-crafted rule thresholds on benign training traffic only
        benign_df = clean_df[clean_df["is_attack"] == 0]
        self.limits = {
            "rate": float(np.percentile(benign_df["rate"], self.pct)),
            "spkts_ratio": float(np.percentile(benign_df["spkts"] / (benign_df["dpkts"] + 1), self.pct)),
            "ct_srv_src": float(np.percentile(benign_df["ct_srv_src"], self.pct)),
            "ct_dst_ltm": float(np.percentile(benign_df["ct_dst_ltm"], self.pct)),
            "burst_density": float(np.percentile(benign_df["spkts"] / (benign_df["dur"] + 0.001), self.pct)),
        }
        return self

    def _get_rules_fired(self, row):
        fired = []
        for name, func in RULE_MAP.items():
            try:
                if bool(func(row, self.limits)):
                    fired.append(name)
            except Exception:
                pass
        return fired

    def predict(self, df, threshold=0.5):
        """Predict alerts for events exceeding the confidence threshold.
        Every alert is validated against schema.json and fails loudly on validation error."""
        if threshold > 1.0:
            return []

        clean_df = df.copy()
        clean_df.columns = [c.strip() for c in clean_df.columns]

        if "event_id" in clean_df.columns:
            event_ids = clean_df["event_id"].to_numpy()
        elif "id" in clean_df.columns:
            event_ids = clean_df["id"].to_numpy()
        else:
            event_ids = np.arange(len(clean_df))

        X = clean_df[self.features]
        probs = self.rf.predict_proba(X)
        if 1 in self.rf.classes_:
            pos_idx = list(self.rf.classes_).index(1)
            scores = probs[:, pos_idx]
        else:
            scores = np.zeros(len(X))

        alerts = []
        qualifying = np.where(scores >= threshold)[0]

        for idx in qualifying:
            score = float(scores[idx])
            row = clean_df.iloc[idx]
            eid = int(event_ids[idx])
            rules_fired = self._get_rules_fired(row)

            top_features = [
                {
                    "feature": str(f),
                    "importance": round(float(self.feature_importances_[f]), 4),
                    "observed_value": float(row[f]),
                }
                for f in self.top_feature_names_
            ]

            alert = {
                "incident_id": f"INC-{eid:08d}",
                "confidence": round(score, 4),
                "contributing_event_ids": [eid],
                "time_window": {
                    "start": "row_index",
                    "end": "row_index",
                },
                "trace": {
                    "model": "random_forest",
                    "rules_fired": rules_fired,
                    "top_features": top_features,
                },
            }

            # Validates against schema.json; fails loudly if invalid
            jsonschema.validate(instance=alert, schema=self._schema)
            alerts.append(alert)

        return alerts

    def save(self, path):
        """Serialize engine state to disk."""
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        state = {
            "n_estimators": self.n_estimators,
            "pct": self.pct,
            "random_state": self.random_state,
            "rf": self.rf,
            "features": self.features,
            "limits": self.limits,
            "feature_importances_": self.feature_importances_,
            "top_feature_names_": self.top_feature_names_,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path):
        """Deserialize engine state from disk."""
        state = joblib.load(path)
        engine = cls(
            n_estimators=state["n_estimators"],
            pct=state["pct"],
            random_state=state["random_state"],
        )
        engine.rf = state["rf"]
        engine.features = state["features"]
        engine.limits = state["limits"]
        engine.feature_importances_ = state["feature_importances_"]
        engine.top_feature_names_ = state["top_feature_names_"]
        return engine


def main():
    parser = argparse.ArgumentParser(description="CorrelationEngine CLI")
    parser.add_argument("--train-csv", help="Path to training CSV")
    parser.add_argument("--input-csv", required=True, help="Path to input event CSV to evaluate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--label-col", default="attack_cat", help="Ground truth column name")
    parser.add_argument("--benign", default="Normal", help="Benign label value")
    parser.add_argument("--model-out", default="model.joblib", help="Path to save trained engine")
    parser.add_argument("--model-in", help="Path to load existing trained engine")
    parser.add_argument("--out", default="alerts.jsonl", help="Output file for generated alerts (.jsonl)")
    args = parser.parse_args()

    if args.model_in and os.path.exists(args.model_in):
        print(f"Loading engine from {args.model_in}...")
        engine = CorrelationEngine.load(args.model_in)
    elif args.train_csv:
        print(f"Training engine on {args.train_csv}...")
        train_df = pd.read_csv(args.train_csv)
        engine = CorrelationEngine()
        engine.fit(train_df, label_col=args.label_col, benign=args.benign)
        if args.model_out:
            engine.save(args.model_out)
            print(f"Engine saved to {args.model_out}")
    else:
        raise ValueError("Either --train-csv or --model-in must be provided.")

    print(f"Reading input events from {args.input_csv}...")
    input_df = pd.read_csv(args.input_csv)
    alerts = engine.predict(input_df, threshold=args.threshold)
    print(f"Generated {len(alerts)} schema-validated alerts at threshold {args.threshold}.")

    parent = os.path.dirname(os.path.abspath(args.out))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for alert in alerts:
            f.write(json.dumps(alert) + "\n")
    print(f"Wrote alerts to {args.out}")


if __name__ == "__main__":
    main()
