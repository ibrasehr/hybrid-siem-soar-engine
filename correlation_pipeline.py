"""
Person A - Correlation engine starter (offline Python prototype).

Covers:
  Pillar 1: rule baseline vs Random Forest + measured throughput/latency
  Pillar 2: drift test (train on some attack families, test on unseen ones)
  Pillar 3: FPR and recall at multiple thresholds
  Pillar 6 (your share): JSON alert with a trace field

Usage:
  pip install pandas numpy scikit-learn
  python correlation_pipeline.py --csv path/to/data.csv --label-col " Label" --benign BENIGN

For CICIDS-2017 use one day's CSV first (e.g. Wednesday or Friday), not all files.
For UNSW-NB15 use UNSW_NB15_training-set.csv with --label-col attack_cat --benign Normal
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split

THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]


def load(csv, label_col, benign):
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    label_col = label_col.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["family"] = df[label_col].astype(str).str.strip()
    df["is_attack"] = (df["family"].str.lower() != benign.lower()).astype(int)
    df["event_id"] = np.arange(len(df))
    exclude = {"label", "id", "attack_cat", "family", "is_attack", "event_id", label_col.lower()}
    num = df.select_dtypes(include=[np.number]).columns
    features = [c for c in num if c.lower() not in exclude]
    print(f"Features ({len(features)}): {features}")
    return df, features


class RuleEngine:
    """Wazuh-style baseline: each rule fires when a feature exceeds the
    99th percentile of benign training traffic. Score = fraction of rules fired."""

    def __init__(self, rule_features, pct=99):
        self.rule_features = rule_features
        self.pct = pct
        self.limits = {}

    def fit(self, X_benign):
        for f in self.rule_features:
            self.limits[f] = np.percentile(X_benign[f], self.pct)
        return self

    def score(self, X):
        fired = np.column_stack([(X[f] > lim).to_numpy() for f, lim in self.limits.items()])
        return fired.mean(axis=1)


def sweep(y_true, scores, name):
    rows = []
    for t in THRESHOLDS:
        pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({
            "model": name, "threshold": t,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
            "precision": precision_score(y_true, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def measure_speed(rf, X, batch_sizes=(100, 1000, 10000)):
    out = []
    for b in batch_sizes:
        if b > len(X):
            continue
        batch = X.iloc[:b]
        t0 = time.perf_counter()
        rf.predict_proba(batch)
        dt = time.perf_counter() - t0
        out.append({"batch": b, "seconds": dt, "events_per_sec": b / dt,
                    "ms_per_event": 1000 * dt / b})
    return pd.DataFrame(out)


def drift_test(df, features, rf_params):
    """Leave-one-family-out: train without family F, test on F + benign."""
    attacks = [f for f in df["family"].unique() if f.lower() != "benign"]
    rows = []
    benign_test = df[df["is_attack"] == 0].sample(frac=0.3, random_state=0)
    benign_train = df[df["is_attack"] == 0].drop(benign_test.index)
    for held in attacks:
        train = pd.concat([benign_train, df[(df["is_attack"] == 1) & (df["family"] != held)]])
        test = pd.concat([benign_test, df[df["family"] == held]])
        if train["is_attack"].nunique() < 2 or len(test) == 0:
            continue
        rf = RandomForestClassifier(**rf_params).fit(train[features], train["is_attack"])
        s = rf.predict_proba(test[features])[:, 1]
        pred = (s >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(test["is_attack"], pred, labels=[0, 1]).ravel()
        rows.append({"held_out_family": held, "n_test_attacks": int(tp + fn),
                     "recall_unseen": tp / (tp + fn) if (tp + fn) else 0.0,
                     "fpr": fp / (fp + tn) if (fp + tn) else 0.0})
    return pd.DataFrame(rows)


def make_alert(rf, features, X_row, event_ids, score, window):
    imp = pd.Series(rf.feature_importances_, index=features).nlargest(5)
    return {
        "incident_id": f"INC-{int(event_ids[0]):08d}",
        "confidence": round(float(score), 4),
        "contributing_event_ids": [int(e) for e in event_ids],
        "time_window": window,
        "trace": {
            "model": "random_forest",
            "top_features": [
                {"feature": f, "importance": round(float(i), 4),
                 "observed_value": float(X_row[f])} for f, i in imp.items()
            ],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label-col", default=" Label")
    ap.add_argument("--benign", default="BENIGN")
    ap.add_argument("--sample", type=int, default=200000, help="max rows to use")
    args = ap.parse_args()

    df, features = load(args.csv, args.label_col, args.benign)
    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
    print(f"Rows: {len(df)}  Features: {len(features)}")
    print(df["family"].value_counts().head(10), "\n")

    train, test = train_test_split(df, test_size=0.3, stratify=df["is_attack"], random_state=42)
    rf_params = dict(n_estimators=100, n_jobs=-1, random_state=42)

    rf = RandomForestClassifier(**rf_params).fit(train[features], train["is_attack"])
    rf_scores = rf.predict_proba(test[features])[:, 1]

    # pick rule features: top 5 by RF importance (replace with hand-chosen rules if you prefer)
    rule_feats = list(pd.Series(rf.feature_importances_, index=features).nlargest(5).index)
    rules = RuleEngine(rule_feats).fit(train[train["is_attack"] == 0][features])
    rule_scores = rules.score(test[features])

    results = pd.concat([sweep(test["is_attack"], rule_scores, "rule_baseline"),
                         sweep(test["is_attack"], rf_scores, "random_forest")])
    print("== Threshold sweep (Pillar 3) ==")
    print(results.round(4).to_string(index=False), "\n")

    print("== Speed on this machine (Pillar 1) ==")
    print(measure_speed(rf, test[features]).round(4).to_string(index=False), "\n")

    print("== Drift: leave-one-family-out (Pillar 2) ==")
    print(drift_test(df, features, rf_params).round(4).to_string(index=False), "\n")

    results.to_csv("threshold_results.csv", index=False)

    # sample alert for the shared schema
    top = test.assign(score=rf_scores).sort_values("score", ascending=False).head(3)
    alert = make_alert(rf, features, top.iloc[0][features], top["event_id"].to_numpy(),
                       top.iloc[0]["score"], {"start": "row_index", "end": "row_index"})
    with open("sample_alert.json", "w") as f:
        json.dump(alert, f, indent=2)
    print("Wrote threshold_results.csv and sample_alert.json")


if __name__ == "__main__":
    main()
