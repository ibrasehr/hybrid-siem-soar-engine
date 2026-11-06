"""Threshold sweep for the response engine (Harshith's results section).

  python -m response_engine.evaluate data/alerts_full.jsonl
Writes results/response_threshold_sweep.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from .adapters import load_alerts
from .assets import AssetRegistry
from .engine import decide_all
from .policy import Policy

DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


def sweep(df: pd.DataFrame, thresholds=None, policy: Optional[Policy] = None,
          registry: Optional[AssetRegistry] = None) -> pd.DataFrame:
    """One row per auto-threshold. Uses decide_all, so it matches the live engine exactly."""
    policy = policy or Policy()
    benign = int((df["true_label"] == 0).sum())
    rows = []
    for t in (thresholds or DEFAULT_THRESHOLDS):
        d = decide_all(df, policy=policy.with_auto_threshold(t), registry=registry,
                       with_explanations=False)
        auto, review, logged = d[d["tier"] >= 2], d[d["tier"] == 1], d[d["tier"] == 0]
        tp = int((auto["true_label"] == 1).sum())
        fp = int((auto["true_label"] == 0).sum())
        rows.append({
            "threshold": t, "alerts": len(d),
            "auto": len(auto), "auto_soft": int((auto["tier"] == 2).sum()),
            "auto_hard": int((auto["tier"] == 3).sum()),
            "auto_rate": len(auto) / len(d) if len(d) else float("nan"),
            "auto_precision": tp / len(auto) if len(auto) else float("nan"),
            "false_auto_actions": fp,                                  # each needs a rollback
            "benign_alerts_auto_actioned": fp / benign if benign else float("nan"),
            "review_queue": len(review),
            "bdi_forced_reviews": int((review["review_reason"] == "bdi_gate").sum()),
            "attacks_waiting_in_review": int((review["true_label"] == 1).sum()),
            "log_only": len(logged),
            "attacks_log_only": int((logged["true_label"] == 1).sum()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alerts", help="alerts_full.jsonl (preferred) or alerts.json")
    ap.add_argument("--out", default="results/response_threshold_sweep.csv")
    ap.add_argument("--no-bdi-gate", action="store_true", help="disable the BDI gate")
    args = ap.parse_args()

    df = load_alerts(args.alerts)
    if len(df) == 100 and Path(args.alerts).name == "alerts.json":
        print("WARNING: this is the 100-alert sample (50 highest-confidence + 50 random).\n"
              "It is not representative. Generate alerts_full.jsonl for reportable numbers.\n")
    policy = Policy(bdi_limit=None) if args.no_bdi_gate else Policy()
    res = sweep(df, policy=policy)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nBusiness Disruption Index uses SIMULATED asset criticality (no host data in UNSW-NB15).")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
