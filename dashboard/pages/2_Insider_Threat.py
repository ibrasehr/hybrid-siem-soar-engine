import streamlit as st
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Insider Threat Analysis", layout="wide")

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

st.title("Insider Threat Detection & Analytics")
st.caption("Behavioral anomaly modeling and risk scoring.")

results_path = ROOT / "results"
if (results_path / "insider_results.csv").exists():
    df_ins = pd.read_csv(results_path / "insider_results.csv")
    st.dataframe(df_ins, width="stretch")
else:
    st.info("Insider threat evaluation metrics and prediction distributions.")
