"""Correlation Engine page (Ibra / Harshith)."""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Correlation Engine", layout="wide")

# Custom Dark & Light Purple Theme Styling
st.markdown("""
<style>
    input[type="range"] {
        accent-color: #a855f7 !important;
    }
    div[data-baseweb="tag"] {
        background-color: #4c1d95 !important;
        color: #f3e8ff !important;
        border: 1px solid #7e22ce !important;
    }
    .stButton button {
        background-color: #7e22ce !important;
        color: #f3e8ff !important;
        border: none;
    }
    .stButton button:hover {
        background-color: #9333ea !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Correlation Engine: rule baseline vs Random Forest")
st.caption("UNSW-NB15, stratified split. Source: results/ in this repo (Ibra).")

results_dir = ROOT / "results"

try:
    # 1. Threshold / Correlation Results & Side-by-Side Line Charts
    threshold_file = results_dir / "threshold_results.csv"
    if threshold_file.exists():
        df_thresh = pd.read_csv(threshold_file)
        
        if "threshold" in df_thresh.columns and "recall" in df_thresh.columns and "model" in df_thresh.columns:
            df_pivot_recall = df_thresh.pivot(index="threshold", columns="model", values="recall")
            df_pivot_fpr = df_thresh.pivot(index="threshold", columns="model", values="fpr") if "fpr" in df_thresh.columns else None

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Recall by threshold")
                st.line_chart(df_pivot_recall)
            with col2:
                if df_pivot_fpr is not None:
                    st.subheader("False-positive rate by threshold")
                    st.line_chart(df_pivot_fpr)

        st.dataframe(df_thresh, width="stretch", height=380)

    # 2. Drift Results Section
    drift_file = results_dir / "drift_results.csv"
    if drift_file.exists():
        st.subheader("Drift: recall on an attack family the model never saw")
        df_drift = pd.read_csv(drift_file)
        if "held_out_family" in df_drift.columns and "recall_unseen" in df_drift.columns:
            st.bar_chart(df_drift.set_index("held_out_family")["recall_unseen"])
        else:
            st.bar_chart(df_drift)
        st.dataframe(df_drift, width="stretch")

    # 3. Speed / Throughput Results Section
    speed_file = results_dir / "speed_results.csv"
    if speed_file.exists():
        st.subheader("Inference throughput (events/sec, scoring only)")
        df_speed = pd.read_csv(speed_file)
        if not df_speed.empty:
            if "batch" in df_speed.columns and "mean_events_per_sec" in df_speed.columns:
                # Pass dataframe with batch as index so x-axis shows 100, 1000, 10000 correctly
                st.bar_chart(df_speed.set_index("batch")[["mean_events_per_sec"]])
            else:
                st.bar_chart(df_speed)
                
            st.caption("Model scoring only; excludes capture, ingestion and feature computation.")

except Exception as e:
    st.error(f"Error loading result files from results/: {e}")