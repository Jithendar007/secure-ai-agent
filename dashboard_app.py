"""
dashboard_app.py
══════════════════════════════════════════════════════════════════════════════
Streamlit Real-Time Security Dashboard

Run with: streamlit run dashboard_app.py

Displays:
  - Current security state with colour-coded indicator
  - Risk score trend chart
  - Per-layer risk breakdown (radar chart)
  - Attack memory cluster table
  - Trust score leaderboard
  - Decision log
  - Red Team simulation panel
══════════════════════════════════════════════════════════════════════════════
"""

import json
import time
import requests
import streamlit as st
import pandas as pd

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="🛡 AI Security Dashboard",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🛡 Secure AI Agent")
st.sidebar.markdown("**Adaptive Stateful Defence System**")
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
if auto_refresh:
    time.sleep(5)
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Red Team Simulator")
scenario = st.sidebar.selectbox(
    "Attack Scenario",
    ["prompt_injection", "jailbreak", "exfil", "escalation", "gradual", "encoding", "indirect"],
)
user_id_rt = st.sidebar.text_input("User ID", value="redteam_bot")
if st.sidebar.button("🔴 Launch Attack"):
    try:
        r = requests.post(f"{API_BASE}/redteam", json={"scenario": scenario, "user_id": user_id_rt})
        result = r.json()
        st.sidebar.success(f"Decision: **{result['decision']}**")
        st.sidebar.json(result)
    except Exception as e:
        st.sidebar.error(f"API unreachable: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("♻️ Reset Security State"):
    try:
        requests.post(f"{API_BASE}/reset")
        st.sidebar.success("State reset to SAFE")
    except Exception as e:
        st.sidebar.error(str(e))

# ─── Fetch Dashboard Data ──────────────────────────────────────────────────────
try:
    data = requests.get(f"{API_BASE}/dashboard", timeout=3).json()
    api_ok = True
except Exception:
    data = {}
    api_ok = False

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🛡 AI Security Dashboard")

if not api_ok:
    st.warning("⚠️ Cannot reach API at localhost:8000. Run `uvicorn main_app:app --reload` first.")
    st.stop()

# ─── State Banner ─────────────────────────────────────────────────────────────
state = data.get("state", "SAFE")
STATE_COLORS = {
    "SAFE":         "#22c55e",
    "SUSPICIOUS":   "#f59e0b",
    "UNDER_ATTACK": "#ef4444",
    "LOCKDOWN":     "#7c3aed",
}
color = STATE_COLORS.get(state, "#888")
st.markdown(
    f"""<div style='background:{color}22;border-left:6px solid {color};
    padding:1rem 1.5rem;border-radius:8px;margin-bottom:1.5rem'>
    <span style='font-size:1.4rem;font-weight:600;color:{color}'>
    Security State: {state}</span>
    <span style='float:right;font-size:0.9rem;color:#888'>
    {data.get("request_count",0)} requests processed</span></div>""",
    unsafe_allow_html=True,
)

# ─── KPI Row ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
re_data = data.get("risk_engine", {})
am_data = data.get("attack_memory", {})

col1.metric("Risk Pressure",    f"{re_data.get('attack_pressure', 0):.3f}")
col2.metric("Recent Avg Risk",  f"{re_data.get('recent_avg', 0):.3f}")
col3.metric("Risk Trend",       f"{re_data.get('trend', 0):.5f}")
col4.metric("Total Attacks",    am_data.get("total_attacks", 0))
col5.metric("Attack Clusters",  am_data.get("clusters", 0))

st.markdown("---")

# ─── Two-column layout ────────────────────────────────────────────────────────
left, right = st.columns([1.4, 1])

with left:
    st.subheader("📜 Decision Log")
    log = data.get("decision_log", [])
    if log:
        df = pd.DataFrame(log)
        if "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
            df = df[["time", "request_id", "decision", "risk_score", "state"]]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No decisions logged yet.")

    st.subheader("⚖️ Risk Engine Weights")
    weights = re_data.get("weights", {})
    if weights:
        w_df = pd.DataFrame([{"layer": k, "weight": round(v, 4)} for k, v in weights.items()])
        st.bar_chart(w_df.set_index("layer")["weight"])

with right:
    st.subheader("🧠 Attack Clusters")
    clusters = data.get("cluster_summary", {})
    if clusters:
        rows = []
        for cid, info in clusters.items():
            rows.append({
                "Cluster": cid,
                "Size":    info["size"],
                "Avg Risk": info["avg_risk"],
                "Tags":    ", ".join(info.get("top_tags", [])),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No attack clusters yet.")

    st.subheader("👤 Trust Scores")
    trust = data.get("trust_scores", {})
    if trust:
        t_df = pd.DataFrame([{"user": k, "trust": round(v, 3)} for k, v in trust.items()])
        st.dataframe(t_df.sort_values("trust"), use_container_width=True)
    else:
        st.info("No users tracked yet.")

    st.subheader("📋 Derived Rules")
    rules = data.get("derived_rules", [])
    for r in rules:
        st.code(r)
    if not rules:
        st.info("Rules generated after 20+ attacks.")

st.markdown("---")

# ─── State Transition Log ──────────────────────────────────────────────────────
st.subheader("🔀 State Transition History")
state_log = data.get("state_log", [])
if state_log:
    sl_df = pd.DataFrame(state_log)
    if "timestamp" in sl_df.columns:
        sl_df["time"] = pd.to_datetime(sl_df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
    st.dataframe(sl_df, use_container_width=True)
else:
    st.info("No state transitions yet.")

# ─── Live Chat Test ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("💬 Live Secure Chat Test")
with st.form("chat_form"):
    test_user = st.text_input("User ID", value="test_user")
    test_msg  = st.text_area("Message", height=80)
    submitted = st.form_submit_button("Send →")
    if submitted and test_msg:
        try:
            resp = requests.post(
                f"{API_BASE}/chat",
                json={"user_id": test_user, "message": test_msg},
                timeout=10
            ).json()
            decision_color = {"ALLOW": "green", "SANITIZE": "orange",
                              "RESTRICT": "red", "BLOCK": "purple"}.get(resp.get("decision",""), "gray")
            st.markdown(
                f"**Decision:** :{decision_color}[{resp.get('decision')}]  |  "
                f"**Risk:** `{resp.get('risk_score')}`  |  "
                f"**State:** `{resp.get('security_state')}`"
            )
            st.text_area("Response", resp.get("output", ""), height=120)
            with st.expander("Audit Trail"):
                for line in resp.get("audit", []):
                    st.code(line)
        except Exception as e:
            st.error(str(e))
