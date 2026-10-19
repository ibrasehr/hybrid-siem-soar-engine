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
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.model_selection import train_test_split


def get_system_specs():
    cpu = "Unknown"
    ram = "Unknown"
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        import platform
        cpu = platform.processor() or "Unknown"

    try:
        import ctypes
        mem = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(mem)):
            ram = f"{mem.value / (1024**2):.2f} GB"
    except Exception:
        pass
    return cpu, ram

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


def rule_high_packet_rate(df, limits):
    """Targets volumetric DoS and flooding attacks by detecting packet transmission rates exceeding benign norms."""
    return df["rate"] > limits["rate"]


def rule_packet_asymmetry(df, limits):
    """Targets scanning, reconnaissance, and unilateral floods where outbound source packets vastly exceed inbound responses."""
    ratio = df["spkts"] / (df["dpkts"] + 1)
    return ratio > limits["spkts_ratio"]


def rule_high_srv_connections(df, limits):
    """Targets service-level port scans and brute-force sweeps that create dense connection counts to the same service."""
    return df["ct_srv_src"] > limits["ct_srv_src"]


def rule_high_dst_density(df, limits):
    """Targets host scanning, probing, and DoS targeting a concentrated destination within a short time window."""
    return df["ct_dst_ltm"] > limits["ct_dst_ltm"]


def rule_short_duration_burst(df, limits):
    """Targets exploit attempts and fast probe bursts delivering high packet volumes over near-zero connection duration."""
    burst = df["spkts"] / (df["dur"] + 0.001)
    return burst > limits["burst_density"]


HAND_CRAFTED_RULES = [
    rule_high_packet_rate,
    rule_packet_asymmetry,
    rule_high_srv_connections,
    rule_high_dst_density,
    rule_short_duration_burst,
]


class RuleEngine:
    """Wazuh-style baseline using 5 hand-crafted behavioral rules fitted on benign training traffic.
    Score = fraction of rules that fired (0.0 to 1.0)."""

    def __init__(self, rules=HAND_CRAFTED_RULES, pct=99):
        self.rules = rules
        self.pct = pct
        self.limits = {}

    def fit(self, df_benign):
        self.limits["rate"] = float(np.percentile(df_benign["rate"], self.pct))
        ratio = df_benign["spkts"] / (df_benign["dpkts"] + 1)
        self.limits["spkts_ratio"] = float(np.percentile(ratio, self.pct))
        self.limits["ct_srv_src"] = float(np.percentile(df_benign["ct_srv_src"], self.pct))
        self.limits["ct_dst_ltm"] = float(np.percentile(df_benign["ct_dst_ltm"], self.pct))
        burst = df_benign["spkts"] / (df_benign["dur"] + 0.001)
        self.limits["burst_density"] = float(np.percentile(burst, self.pct))
        return self

    def score(self, df):
        fired = np.column_stack([rule(df, self.limits).to_numpy() for rule in self.rules])
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


def measure_speed(rf, X, batch_sizes=(100, 1000, 10000), runs=5):
    out = []
    for b in batch_sizes:
        if b > len(X):
            continue
        batch = X.iloc[:b]
        speeds = []
        for _ in range(runs):
            t0 = time.perf_counter()
            rf.predict_proba(batch)
            dt = time.perf_counter() - t0
            speeds.append(b / dt)
        out.append({
            "batch": b,
            "mean_events_per_sec": float(np.mean(speeds)),
            "min_events_per_sec": float(np.min(speeds)),
            "max_events_per_sec": float(np.max(speeds)),
        })
    return pd.DataFrame(out)


def drift_test(df, features, rf_params, benign="BENIGN"):
    """Leave-one-family-out: train without family F, test on F + benign."""
    attacks = [f for f in df["family"].unique() if f.lower() != benign.lower()]
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


def make_alert(rf, features, X_row, event_ids, score, window, rules=None):
    imp = pd.Series(rf.feature_importances_, index=features).nlargest(5)
    rules_fired = []
    if rules is not None:
        for rule in rules.rules:
            try:
                if bool(rule(X_row, rules.limits)):
                    rules_fired.append(rule.__name__)
            except Exception:
                pass
    alert = {
        "incident_id": f"INC-{int(event_ids[0]):08d}",
        "confidence": round(float(score), 4),
        "contributing_event_ids": [int(e) for e in event_ids],
        "time_window": window,
        "trace": {
            "model": "random_forest",
            "rules_fired": rules_fired,
            "top_features": [
                {"feature": f, "importance": round(float(i), 4),
                 "observed_value": float(X_row[f])} for f, i in imp.items()
            ],
        },
    }
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")
    if os.path.exists(schema_path):
        import jsonschema
        with open(schema_path, "r", encoding="utf-8") as f:
            jsonschema.validate(instance=alert, schema=json.load(f))
    return alert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label-col", default=" Label")
    ap.add_argument("--benign", default="BENIGN")
    ap.add_argument("--sample", type=int, default=200000, help="max rows to use")
    ap.add_argument("--ablate", default="", help="comma-separated feature names to drop before training")
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds for train/test split evaluation")
    args = ap.parse_args()

    df, features = load(args.csv, args.label_col, args.benign)
    if args.ablate:
        to_drop = [f.strip() for f in args.ablate.split(",") if f.strip()]
        features = [f for f in features if f not in to_drop]
        print(f"Ablated features: {to_drop}")
        print(f"Remaining features ({len(features)}): {features}\n")

    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
    print(f"Rows: {len(df)}  Features: {len(features)}")
    print(df["family"].value_counts().head(10), "\n")

    rf_params = dict(n_estimators=100, n_jobs=-1)

    seed_sweeps = []
    base_rf = None
    base_test = None
    base_rf_scores = None
    base_rules = None

    for i in range(args.seeds):
        seed = 42 + i
        s_train, s_test = train_test_split(df, test_size=0.3, stratify=df["is_attack"], random_state=seed)
        s_rf = RandomForestClassifier(**dict(rf_params, random_state=seed)).fit(s_train[features], s_train["is_attack"])
        s_rf_scores = s_rf.predict_proba(s_test[features])[:, 1]

        s_rules = RuleEngine().fit(s_train[s_train["is_attack"] == 0])
        s_rule_scores = s_rules.score(s_test)

        s_res = pd.concat([sweep(s_test["is_attack"], s_rule_scores, "rule_baseline"),
                           sweep(s_test["is_attack"], s_rf_scores, "random_forest")])
        s_res["seed"] = seed
        seed_sweeps.append(s_res)

        if i == 0:
            base_rf = s_rf
            base_test = s_test
            base_rf_scores = s_rf_scores
            base_rules = s_rules

    all_seeds_df = pd.concat(seed_sweeps, ignore_index=True)
    multiseed_summary = (
        all_seeds_df.groupby(["model", "threshold"])
        .agg(
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            fpr_mean=("fpr", "mean"),
            fpr_std=("fpr", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
        )
        .reset_index()
    )

    print(f"== Multi-seed Threshold Sweep ({args.seeds} seeds) (Pillar 3) ==")
    print(multiseed_summary.round(4).to_string(index=False), "\n")

    print("== Speed on this machine (Pillar 1) ==")
    cpu, ram = get_system_specs()
    print(f"CPU Model: {cpu}")
    print(f"RAM: {ram}")
    speed_df = measure_speed(base_rf, base_test[features])
    print(speed_df.round(4).to_string(index=False), "\n")

    print("== Drift: leave-one-family-out (Pillar 2) ==")
    drift_df = drift_test(df, features, dict(rf_params, random_state=42), args.benign)
    print(drift_df.round(4).to_string(index=False), "\n")

    os.makedirs("results", exist_ok=True)
    prefix = "ablation_" if args.ablate else ""
    multiseed_summary.to_csv(f"results/{prefix}threshold_multiseed.csv", index=False)
    single_seed_results = all_seeds_df[all_seeds_df["seed"] == 42].drop(columns=["seed"])
    single_seed_results.to_csv(f"results/{prefix}threshold_results.csv", index=False)
    drift_df.to_csv(f"results/{prefix}drift_results.csv", index=False)
    speed_df.to_csv(f"results/{prefix}speed_results.csv", index=False)
    multiseed_summary.to_csv("threshold_results.csv", index=False)

    # sample alert for the shared schema
    top = base_test.assign(score=base_rf_scores).sort_values("score", ascending=False).head(3)
    alert = make_alert(base_rf, features, top.iloc[0], top["event_id"].to_numpy(),
                       top.iloc[0]["score"], {"start": "row_index", "end": "row_index"},
                       rules=base_rules)
    with open("sample_alert.json", "w") as f:
        json.dump(alert, f, indent=2)
    print(f"Wrote results/{prefix}threshold_multiseed.csv, results/{prefix}threshold_results.csv, results/{prefix}drift_results.csv, results/{prefix}speed_results.csv, and sample_alert.json")


if __name__ == "__main__":
    main()
