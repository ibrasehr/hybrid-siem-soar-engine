"""Governed SOAR Response Engine Dashboard"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from response_engine import (AssetRegistry, Policy, audit_jsonl, decide_all, load_alerts,
                             load_insider_csv, run_pipeline, suggest_threshold)

st.set_page_config(page_title="AI-Driven SIEM/SOAR", layout="wide", initial_sidebar_state="expanded")

# Bulletproof CSS with smooth rounded corners and deep purple/lavender theme
st.markdown("""
<style>
    /* Force deep purple background on outer and ALL nested inner elements of alert/info boxes */
    div.stAlert, div.stInfo, div[data-baseweb="notification"], div[data-testid="stNotification"],
    div.stAlert *, div.stInfo *, div[data-baseweb="notification"] *, div[data-testid="stNotification"] * {
        background-color: #3b1f5e !important;
        background-image: none !important;
    }
    
    /* Smooth rounded corners, border outline, and accent bar */
    div.stAlert, div.stInfo, div[data-baseweb="notification"], div[data-testid="stNotification"] {
        border: 1px solid #a855f7 !important;
        border-left: 5px solid #c084fc !important;
        border-radius: 12px !important; /* Smooth rounded corners */
        overflow: hidden !important;     /* Keeps inner background within rounded corners */
    }
    
    /* Force all text paragraphs and spans inside to bright light lavender */
    div.stAlert p, div.stInfo p, div[data-baseweb="notification"] p, div[data-testid="stNotification"] p,
    div.stAlert span, div.stInfo span, div[data-baseweb="notification"] span, div[data-testid="stNotification"] span {
        color: #f3e8ff !important;
    }

    /* Force icons to match lavender */
    div.stAlert svg, div.stInfo svg, div[data-baseweb="notification"] svg, div[data-testid="stNotification"] svg {
        fill: #f3e8ff !important;
        color: #f3e8ff !important;
    }

    /* Custom Dark & Light Purple Theme Styling for inputs and buttons */
    input[type="range"] {
        accent-color: #a855f7 !important;
    }
    div[data-baseweb="tag"] {
        background-color: #4c1d95 !important;
        color: #f3e8ff !important;
        border: 1px solid #7e22ce !important;
        border-radius: 6px !important;
    }
    .stButton button {
        background-color: #7e22ce !important;
        color: #f3e8ff !important;
        border: none;
        border-radius: 8px !important;
    }
    .stButton button:hover {
        background-color: #9333ea !important;
    }
</style>
""", unsafe_allow_html=True)

def main_dashboard():
    st.title("Governed SOAR Response Engine")
    st.caption("Four tiers (log / review / auto-soft / auto-hard), BDI gate, analyst queue, "
               "verified rollback, feedback log.")

    # ---------- data ----------
    CANDIDATES = [ROOT / "data" / "alerts_full.jsonl", ROOT / "alerts_full.jsonl", ROOT / "alerts.json"]
    found = [p for p in CANDIDATES if p.exists()]

    @st.cache_data
    def _load(path_str, mtime):
        return load_alerts(path_str)

    uploaded = st.sidebar.file_uploader("Upload alerts (.json / .jsonl)", type=["json", "jsonl"])
    if uploaded is not None:
        corr, label = load_alerts(uploaded), uploaded.name
    elif found:
        choice = st.sidebar.selectbox("Alert file", [str(p.relative_to(ROOT)) for p in found])
        corr, label = _load(str(ROOT / choice), (ROOT / choice).stat().st_mtime), choice
    else:
        st.error("No alerts found. Put alerts_full.jsonl in data/ or upload one in the sidebar.")
        st.stop()

    frames = [corr]
    if (ROOT / "data" / "insider_predictions.csv").exists():
        frames.append(load_insider_csv(ROOT / "data" / "insider_predictions.csv"))
    df = pd.concat(frames, ignore_index=True)

    if label.endswith("alerts.json") and len(corr) == 100:
        st.warning("Using the 100-alert sample (50 highest-confidence + 50 random). It is biased toward "
                   "high confidence, so don't report numbers from it. Generate alerts_full.jsonl.")

    # ---------- policy ----------
    st.sidebar.subheader("Policy")
    auto_t = st.sidebar.slider("Auto-response threshold", 0.50, 0.99, 0.90, 0.01)
    hard_t = max(st.sidebar.slider("Hard-containment threshold", 0.50, 1.00, 0.97, 0.01), auto_t)
    gate_on = st.sidebar.checkbox("BDI gate (hold high-disruption actions for approval)", True)
    bdi_limit = st.sidebar.slider("BDI limit (criticality x severity)", 1, 25, 12, disabled=not gate_on)
    capacity = st.sidebar.number_input("Analyst queue capacity", 1, 1000, 50)
    policy = Policy(review_floor=min(0.50, auto_t), auto_threshold=auto_t, hard_threshold=hard_t,
                    bdi_limit=float(bdi_limit) if gate_on else None, queue_capacity=int(capacity))
    registry = AssetRegistry()

    for key in ("approved", "rejected", "reverted"):
        st.session_state.setdefault(key, set())

    pre = decide_all(df, policy=policy, registry=registry, with_explanations=False)
    review_ids = set(pre.loc[pre["tier"] == 1, "id"])
    approved = st.session_state.approved & review_ids
    rejected = st.session_state.rejected & review_ids
    executed = set(pre.loc[pre["tier"] >= 2, "id"]) | approved
    reverted = st.session_state.reverted & executed

    res = run_pipeline(df, policy, registry, approve_ids=sorted(approved), reject_ids=sorted(rejected),
                       revert_ids=sorted(reverted))
    dec = res.decisions
    auto = dec[dec["tier"] >= 2]
    review = dec[dec["tier"] == 1]

    c = st.columns(6)
    c[0].metric("Alerts", f"{len(dec):,}")
    c[1].metric("Auto-soft", int((dec["tier"] == 2).sum()))
    c[2].metric("Auto-hard", int((dec["tier"] == 3).sum()))
    c[3].metric("Human review", f"{len(review):,}",
                help=f"{res.queue.overflow_count} beyond the queue capacity of {policy.queue_capacity}")
    c[4].metric("False auto-actions", int((auto["true_label"] == 0).sum()),
                help="Auto-actions on benign traffic. Each one needs a rollback.")
    c[5].metric("Attacks waiting in review", int((review["true_label"] == 1).sum()))
    st.caption(f"Source: {label}. Ground truth feeds only the evaluation counters, never a decision. "
               + ("Asset criticality is SIMULATED (UNSW-NB15 has no host data)." if registry.simulated else ""))

    t1, t2, t3, t4, t5 = st.tabs(["Decision log", "Review queue", "Rollback ledger", "Feedback", "Audit"])

    with t1:
        pick = st.multiselect("Tier", ["LOG_ONLY", "REVIEW", "AUTO_SOFT", "AUTO_HARD"],
                              ["REVIEW", "AUTO_SOFT", "AUTO_HARD"])
        show = ["id", "target", "confidence", "tier_name", "action", "bdi", "review_reason", "explanation"]
        st.dataframe(dec[dec["tier_name"].isin(pick)][show].head(500), width="stretch", height=380)
        st.download_button("Download decision log (CSV)", dec[show].to_csv(index=False), "decision_log.csv")

    with t2:
        q = res.queue
        done = approved | rejected
        st.write(f"{len(q.queue)} in the analyst queue, {q.overflow_count} waiting beyond capacity "
                 f"(never dropped). Highest confidence first.")
        pending = q.queue[~q.queue["id"].isin(done)]
        if len(pending):
            pick_id = st.selectbox("Select alert", pending["id"])
            row = pending[pending["id"] == pick_id].iloc[0]
            st.info(row["explanation"])
            st.write(f"Proposed action: **{row['proposed_action']}** (BDI {row['bdi']}, "
                     f"criticality {row['asset_criticality']})")
            if row["top_features"]:
                st.dataframe(pd.DataFrame(row["top_features"]), width="stretch")
            a, b = st.columns(2)
            if a.button("Approve and execute"):
                st.session_state.approved.add(pick_id); st.rerun()
            if b.button("Reject (false alarm)"):
                st.session_state.rejected.add(pick_id); st.rerun()

    with t3:
        led = res.ledger.to_frame()
        st.dataframe(led.head(500), width="stretch", height=300)
        active = res.ledger.active_ids()
        if active:
            pick_id = st.selectbox("Select executed action", active[:200], key="rb")
            if st.button("Mark false positive and revert"):
                st.session_state.reverted.add(pick_id); st.rerun()

    with t4:
        fb = res.feedback.to_frame()
        st.dataframe(fb, width="stretch")
        s = suggest_threshold(fb)
        st.write(f"Threshold suggestion: **{s['suggested_threshold']}** ({s['reason']}). "
                 "Suggestion only; it is never applied automatically.")
        st.download_button("Download feedback_log.csv", fb.to_csv(index=False), "feedback_log.csv")

    with t5:
        st.caption("One structured record per decision (no ground truth).")
        st.download_button("Download audit records (JSONL)", audit_jsonl(dec, policy, registry),
                           "audit_records.jsonl")

# Define native navigation structure with icons
pg = st.navigation({
    "Overview": [
        st.Page(main_dashboard, title="SOAR Dashboard"),
    ],
    "Analytics": [
        st.Page("pages/1_Correlation.py", title="Event Correlation"),
        st.Page("pages/2_Insider_Threat.py", title="Insider Threat"),
    ]
})

pg.run()