"""Insider Threat Detection and Analytics page."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="Insider Threat Analysis", layout="wide")

st.markdown("""
<style>
    input[type="range"] { accent-color: #a855f7 !important; }
    div[data-baseweb="tag"] {
        background-color: #4c1d95 !important;
        color: #f3e8ff !important;
        border: 1px solid #7e22ce !important;
    }
    .stButton button {
        background-color: #7e22ce !important;
        color: #f3e8ff !important;
        border: none;
        border-radius: 8px !important;
    }
    .stButton button:hover { background-color: #9333ea !important; }
</style>
""", unsafe_allow_html=True)

st.title("Insider Threat Detection & Analytics")
st.caption("Random Forest employee-behaviour scoring combined with rule-based detection.")

ALERTS_PATH = DATA_DIR / "insider_threat_alerts.json"
PREDICTIONS_PATH = DATA_DIR / "insider_threat_predictions.csv"

def load_saved_alerts():
    if not ALERTS_PATH.exists():
        return pd.DataFrame()
    with ALERTS_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    if df.empty:
        return df
    for col, default in {
        "user": "Unknown", "source": "RandomForest", "confidence": 0.0,
        "predicted_label": 0, "rule_alert": 0, "rule_reasons": [], "severity": "Low"
    }.items():
        if col not in df.columns:
            df[col] = default
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0).clip(0, 1)
    df["predicted_label"] = pd.to_numeric(df["predicted_label"], errors="coerce").fillna(0).astype(int)
    df["rule_alert"] = pd.to_numeric(df["rule_alert"], errors="coerce").fillna(0).astype(int)
    return df

saved_alerts = load_saved_alerts()

st.subheader("Detected insider-threat alerts")
if saved_alerts.empty:
    st.info("No saved insider-threat alerts found. Check data/insider_threat_alerts.json.")
else:
    total = len(saved_alerts)
    ml_threats = int((saved_alerts["predicted_label"] == 1).sum())
    rule_alerts = int((saved_alerts["rule_alert"] == 1).sum())
    high_alerts = int((saved_alerts["severity"].astype(str).str.lower() == "high").sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total alerts", total)
    m2.metric("ML threat predictions", ml_threats)
    m3.metric("Behavioural rule alerts", rule_alerts)
    m4.metric("High severity", high_alerts)

    f1, f2 = st.columns([1, 2])
    severity_options = ["All", "High", "Medium", "Low"]
    severity_filter = f1.selectbox("Filter by severity", severity_options)
    search_user = f2.text_input("Search employee/user ID")
    view = saved_alerts.copy()
    if severity_filter != "All":
        view = view[view["severity"].astype(str).str.lower() == severity_filter.lower()]
    if search_user.strip():
        view = view[view["user"].astype(str).str.contains(search_user.strip(), case=False, na=False)]

    display = view.copy()
    display["ML result"] = display["predicted_label"].map({1: "Threat", 0: "Normal"})
    display["Rule result"] = display["rule_alert"].map({1: "Flagged", 0: "Not flagged"})
    display["Confidence"] = (display["confidence"] * 100).round(1).astype(str) + "%"
    display["Rule reasons"] = display["rule_reasons"].apply(
        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
    )
    st.dataframe(
        display[["user", "severity", "ML result", "Confidence", "Rule result", "Rule reasons"]],
        width="stretch", hide_index=True
    )
    st.download_button(
        "Download filtered insider alerts (CSV)",
        display[["user", "source", "confidence", "predicted_label", "rule_alert", "rule_reasons", "severity"]].to_csv(index=False),
        "insider_threat_alerts.csv",
        mime="text/csv"
    )

    with st.expander("View individual alert details"):
        for _, alert in view.iterrows():
            label = f'{alert["user"]} — {alert["severity"]} severity'
            with st.expander(label):
                st.write(f'ML prediction: **{"Threat" if alert["predicted_label"] == 1 else "Normal"}**')
                st.write(f'Random Forest confidence: **{alert["confidence"]:.1%}**')
                st.write(f'Behavioural rule alert: **{"Yes" if alert["rule_alert"] else "No"}**')
                reasons = alert["rule_reasons"] if isinstance(alert["rule_reasons"], list) else []
                if reasons:
                    st.write("Reasons:")
                    for reason in reasons:
                        st.write(f"- {reason}")
                else:
                    st.write("No behavioural rules fired.")

st.divider()
st.subheader("Run detection on new employee activity")
st.write("Upload an event CSV with these columns: `event_id`, `timestamp`, `user`, and `event_type`.")
uploaded_events = st.file_uploader("Upload employee activity CSV", type=["csv"], key="insider_events")

if uploaded_events is not None:
    try:
        events = pd.read_csv(uploaded_events)

        # Accept both the engine's standard column names and the CERT
        # preprocessed dataset's original names (id + file).
        if "event_id" not in events.columns and "id" in events.columns:
            events = events.rename(columns={"id": "event_id"})
        if "event_type" not in events.columns and "file" in events.columns:
            events = events.rename(columns={"file": "event_type"})

        required = {"event_id", "timestamp", "user", "event_type"}
        missing = sorted(required - set(events.columns))
        if missing:
            st.error("CSV is missing required columns after CERT format mapping: " + ", ".join(missing))
            st.info("Accepted formats: event_id/timestamp/user/event_type, or CERT format id/timestamp/user/file.")
        else:
            # The supplied CERT comparison CSV contains mixed/shifted rows.
            # Keep only rows whose event source is a known activity category.
            valid_types = {"http", "device", "email", "logon", "file"}
            before_filter = len(events)
            events["event_type"] = events["event_type"].astype(str).str.strip().str.lower()
            events = events[events["event_type"].isin(valid_types)].copy()
            events["timestamp"] = pd.to_datetime(events["timestamp"], format="mixed", errors="coerce")
            events = events.dropna(subset=["event_id", "timestamp", "user"])
            skipped_rows = before_filter - len(events)
            if events.empty:
                st.error("No valid activity rows were found. Check that the CSV contains valid CERT activity records.")
            else:
                st.caption(f"Loaded {len(events):,} valid activity records across {events['user'].nunique():,} employees.")
                if skipped_rows:
                    st.warning(f"Skipped {skipped_rows:,} malformed or unsupported rows in the uploaded CSV. The original file was not changed.")
            if not events.empty and st.button("Run Insider Threat Detection", type="primary"):
                try:
                    from insider_threat_engine import InsiderThreatEngine
                    engine = InsiderThreatEngine(
                        model_path=str(MODEL_DIR / "insider_threat_model.pkl"),
                        features_path=str(MODEL_DIR / "insider_threat_features.pkl"),
                    )
                    predictions = engine.predict(events)
                    predictions["severity"] = predictions.apply(
                        lambda r: "High" if r["predicted_label"] == 1 and r["confidence"] >= 0.80
                        else ("Medium" if r["predicted_label"] == 1 or r["rule_alert"] == 1 else "Low"),
                        axis=1
                    )
                    predictions["ML result"] = predictions["predicted_label"].map({1: "Threat", 0: "Normal"})
                    predictions["Confidence"] = (predictions["confidence"] * 100).round(1).astype(str) + "%"
                    predictions["Rule result"] = predictions["rule_alert"].map({1: "Flagged", 0: "Not flagged"})
                    predictions["Rule reasons"] = predictions["rule_reasons"].apply(
                        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
                    )
                    st.success(f"Detection completed for {len(predictions)} employees.")
                    st.dataframe(
                        predictions[["user", "severity", "ML result", "Confidence", "Rule result", "Rule reasons"]],
                        width="stretch", hide_index=True
                    )
                    st.download_button(
                        "Download new detection results (CSV)",
                        predictions.to_csv(index=False),
                        "new_insider_threat_predictions.csv",
                        mime="text/csv"
                    )
                except Exception as exc:
                    st.error(f"Could not run the saved model: {exc}")
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")

st.divider()
st.subheader("Model evaluation sample")
st.caption(
    "The file below contains the held-out evaluation sample prepared during model testing. "
    "Its true_label column is for offline evaluation only; it is not used as a live alert."
)
if PREDICTIONS_PATH.exists():
    eval_df = pd.read_csv(PREDICTIONS_PATH)
    st.dataframe(eval_df.drop(columns=["true_label"], errors="ignore"), width="stretch", hide_index=True)
    st.download_button(
        "Download prediction sample (CSV)",
        eval_df.drop(columns=["true_label"], errors="ignore").to_csv(index=False),
        "insider_threat_prediction_sample.csv",
        mime="text/csv"
    )
