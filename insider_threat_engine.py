
import pandas as pd
import joblib
from pathlib import Path


class InsiderThreatEngine:

    def __init__(self, model_path=None, features_path=None):
        base_dir = Path(__file__).resolve().parent
        model_path = model_path or (base_dir / "models" / "insider_threat_model.pkl")
        features_path = features_path or (base_dir / "models" / "insider_threat_features.pkl")

        self.model = joblib.load(model_path)
        self.feature_columns = joblib.load(features_path)

    def build_features(self, events):

        data = events.copy()

        data["timestamp"] = pd.to_datetime(
            data["timestamp"], format="mixed", errors="coerce"
        )

        data["hour"] = data["timestamp"].dt.hour
        data["day_of_week"] = data["timestamp"].dt.dayofweek

        data["after_hours"] = (
            (data["hour"] < 8) | (data["hour"] >= 18)
        ).astype(int)

        data["weekend"] = (
            data["day_of_week"] >= 5
        ).astype(int)

        return data.groupby("user").agg(
            total_events=("event_id", "count"),
            http_count=("event_type", lambda x: (x == "http").sum()),
            device_count=("event_type", lambda x: (x == "device").sum()),
            email_count=("event_type", lambda x: (x == "email").sum()),
            logon_count=("event_type", lambda x: (x == "logon").sum()),
            file_count=("event_type", lambda x: (x == "file").sum()),
            after_hours_count=("after_hours", "sum"),
            weekend_count=("weekend", "sum"),
            active_days=("timestamp", lambda x: x.dt.date.nunique())
        ).reset_index()

    def apply_rules(self, features):

        results = []

        for _, row in features.iterrows():

            reasons = []

            if row["total_events"] >= 100:
                reasons.append("High event volume")

            if row["http_count"] >= 100:
                reasons.append("Excessive web activity")

            if row["device_count"] >= 5:
                reasons.append("Unusual device usage")

            if row["file_count"] >= 3:
                reasons.append("Excessive file activity")

            if row["after_hours_count"] > 6:
                reasons.append("Unusual after-hours activity")

            results.append({
                "rule_alert": int(len(reasons) > 0),
                "rule_reasons": reasons
            })

        return pd.DataFrame(results)

    def predict(self, events):

        features = self.build_features(events)

        if features.empty:
            return pd.DataFrame(columns=[
                "user",
                "source",
                "confidence",
                "predicted_label",
                "rule_alert",
                "rule_reasons"
            ])

        X = features[self.feature_columns]

        probabilities = self.model.predict_proba(X)
        threat_index = list(self.model.classes_).index(1)

        features["confidence"] = probabilities[:, threat_index]
        features["predicted_label"] = self.model.predict(X)
        features["source"] = "insider_threat"

        rules = self.apply_rules(features)
        features = pd.concat(
            [features.reset_index(drop=True), rules], axis=1
        )

        return features[[
            "user",
            "source",
            "confidence",
            "predicted_label",
            "rule_alert",
            "rule_reasons"
        ]]
