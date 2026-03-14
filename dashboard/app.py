"""
IPL Match Predictor
Run: python3 -m streamlit run dashboard/app.py
"""

import streamlit as st
import pickle
import os
import sys
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.pressure_index import live_collapse_probability
from models.unified_predictor import predict_match
from models.toss_alpha_decay import get_toss_edge_feature

st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏", layout="wide", initial_sidebar_state="expanded")

# Theme: strong text contrast so labels are always readable
st.markdown("""
<style>
    .stApp { background: #f8fafc; }
    main { padding: 2rem 3rem 4rem; max-width: 900px; }
    /* Sidebar: force dark text so labels are readable */
    [data-testid="stSidebar"] { background: #ffffff !important; }
    [data-testid="stSidebar"] label { color: #1e293b !important; font-weight: 600; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #334155 !important; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #0f172a !important; font-weight: 600; }
    [data-testid="stSidebar"] .stCaptionContainer label { color: #475569 !important; font-weight: 500; }
    [data-testid="block-container"] { padding: 0; }
    /* Main content: dark headings and body */
    h1, h2, h3 { color: #0f172a !important; font-weight: 600; }
    p, .stMarkdown { color: #1e293b !important; }
    .stCaptionContainer label { color: #475569 !important; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), "..")
    try:
        with open(os.path.join(base, "models/saved/lgb_pre.pkl"), "rb") as f: lgb_pre = pickle.load(f)
        with open(os.path.join(base, "models/saved/lgb_in.pkl"), "rb") as f: lgb_in = pickle.load(f)
        with open(os.path.join(base, "models/saved/meta.pkl"), "rb") as f: meta = pickle.load(f)
        with open(os.path.join(base, "models/saved/toss_alpha.pkl"), "rb") as f: toss_df = pickle.load(f)
        with open(os.path.join(base, "models/saved/collapse_model.pkl"), "rb") as f: collapse_model = pickle.load(f)
        with open(os.path.join(base, "models/saved/collapse_cal.pkl"), "rb") as f: collapse_cal = pickle.load(f)
    except FileNotFoundError:
        return None, None, None, None, None, None, None, None
    pre_calibrator = team_stats = None
    try:
        with open(os.path.join(base, "models/saved/pre_calibrator.pkl"), "rb") as f: pre_calibrator = pickle.load(f)
    except FileNotFoundError: pass
    try:
        with open(os.path.join(base, "models/saved/team_stats.pkl"), "rb") as f: team_stats = pickle.load(f)
    except FileNotFoundError: pass
    return lgb_pre, lgb_in, meta, toss_df, collapse_model, collapse_cal, pre_calibrator, team_stats

lgb_pre, lgb_in, meta, toss_df, collapse_model, collapse_cal, pre_calibrator, team_stats = load_models()
if lgb_pre is None:
    st.error("Models not found. Run: `python3 run_from_kaggle.py`")
    st.stop()

# --- Header ---
st.title("🏏 IPL Match Predictor")
st.markdown("Predict the winner — before the match or as the chase unfolds.")
st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.header("Match setup")
    team_a = st.selectbox("Batting first", ["Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore", "Kolkata Knight Riders", "Rajasthan Royals", "Delhi Capitals", "Sunrisers Hyderabad", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants"], key="team_a")
    team_b = st.selectbox("Chasing", ["Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore", "Kolkata Knight Riders", "Rajasthan Royals", "Delhi Capitals", "Sunrisers Hyderabad", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants"], key="team_b")
    venue_options = ["Wankhede Stadium", "MA Chidambaram Stadium", "Eden Gardens", "M Chinnaswamy Stadium", "Arun Jaitley Stadium", "Rajiv Gandhi International Stadium", "Sawai Mansingh Stadium", "Narendra Modi Stadium", "Dubai International Cricket Stadium", "Sharjah Cricket Stadium", "Sheikh Zayed Stadium", "Brabourne Stadium"]
    if toss_df is not None and hasattr(toss_df, "venue"):
        venue_options = list(dict.fromkeys(venue_options + sorted(toss_df["venue"].dropna().unique().tolist())[:20]))
    venue = st.selectbox("Venue", venue_options, key="venue")
    st.divider()
    st.subheader("Toss")
    toss_winner = st.radio("Toss won by", [team_a, team_b], key="toss_winner")
    toss_dec = st.radio("Decision", ["bat", "field"], key="toss_dec", horizontal=True)
    is_playoff = st.checkbox("Playoff or final", value=False, key="playoff")
    st.divider()
    st.subheader("Live (optional)")
    st.caption("For in-play prediction. Leave at 0 for pre-match.")
    current_over = st.slider("Over", 0, 20, 0, key="over")
    score_1st = st.number_input("1st innings total", 0, 300, 150, key="score_1")
    score_2nd = st.number_input("2nd innings score", 0, 300, 0, key="score_2")
    wkts_2nd = st.slider("Wickets lost", 0, 10, 0, key="wkts_2")
    batter_sr = st.slider("Batter strike rate", 0, 250, 120, key="batter_sr")
    partnership = st.slider("Partnership balls", 0, 60, 6, key="partnership")
    bowler_eco = st.slider("Bowler economy", 4.0, 18.0, 8.0, 0.5, key="bowler_eco")

# --- Features & prediction ---
ERA_FUTURE = "modern"
toss_edge_raw = get_toss_edge_feature(toss_df, venue, toss_dec, ERA_FUTURE) if toss_df is not None else 0.0
toss_edge_score = toss_edge_raw if toss_winner == team_a else -toss_edge_raw

def _stat(team, key, default):
    return (team_stats or {}).get(team, {}).get(key, default)

pre_feats = {
    "toss_edge_score": float(toss_edge_score),
    "elo_delta": float(_stat(team_a, "elo", 1500) - _stat(team_b, "elo", 1500)),
    "form_delta": float(_stat(team_a, "form", 0.5) - _stat(team_b, "form", 0.5)),
    "h2h_venue_wr": 0.5, "is_playoff": 1 if is_playoff else 0, "match_number_norm": 0.5,
}

use_live = current_over > 0 and score_1st > 0
in_feats = None
if use_live and lgb_in is not None:
    runs_needed = max(0, score_1st + 1 - score_2nd)
    balls_remaining = max(1, (20 - current_over) * 6)
    balls_done = current_over * 6
    rrr = (runs_needed * 6) / balls_remaining
    crr = (score_2nd * 6) / balls_done if balls_done else rrr
    phase_pct = balls_done / 120.0
    rrr_gap_norm = max(0, min(1, (rrr - crr + 4) / 16))
    wkt_p = (wkts_2nd / 10) * (1 + phase_pct * 0.5)
    form_p = max(0, min(1, 1 - (batter_sr - 40) / 260)) if batter_sr <= 300 else 0
    part_p = 1 - min(partnership, 60) / 60
    bowl_n = 1 - (min(max(bowler_eco, 4), 14) - 4) / 10
    pressure_index = max(0, min(1, 0.30 * rrr_gap_norm + 0.25 * wkt_p + 0.15 * phase_pct + 0.15 * form_p + 0.10 * part_p + 0.05 * bowl_n))
    in_feats = {"rrr": rrr, "crr": crr, "rrr_gap": rrr - crr, "rrr_ratio": rrr / crr if crr > 0 else 1.0, "wickets_fallen": wkts_2nd, "wickets_remaining": 10 - wkts_2nd, "phase_pct": phase_pct, "is_powerplay": 1 if current_over < 6 else 0, "is_death": 1 if current_over >= 16 else 0, "batter_sr_recent": float(batter_sr), "partnership_ball": partnership, "bowler_economy_live": bowler_eco, "pressure_index": pressure_index, "runs_needed": runs_needed, "balls_remaining": balls_remaining}

pred = predict_match(lgb_pre, lgb_in, meta, pre_feats, in_feats, pre_calibrator=pre_calibrator)
win_a = (pred.get("win_prob_team_a") or 0.5) * 100
win_b = (pred.get("win_prob_team_b") or 0.5) * 100

# --- Main: Win probability (Plotly bar) ---
st.subheader("Win probability")
fig = go.Figure(go.Bar(
    x=[win_a, win_b],
    y=[team_a, team_b],
    orientation="h",
    marker_color=["#2563eb", "#dc2626"],
    text=[f"{win_a:.0f}%", f"{win_b:.0f}%"],
    textposition="inside",
    textfont=dict(size=16, color="white", family="sans-serif"),
))
fig.update_layout(
    height=140,
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, tickfont=dict(size=14, color="#1e293b", family="sans-serif")),
    font=dict(color="#1e293b", size=14),
    showlegend=False,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

fav = team_a if win_a >= 50 else team_b
st.markdown(f"**Favourite:** {fav} ({max(win_a, win_b):.0f}%) · *{'Live' if use_live else 'Pre-match'} prediction*")
st.markdown("---")

# --- Two cards: Collapse risk + Toss ---
col1, col2 = st.columns(2)
with col1:
    with st.container():
        st.subheader("Collapse risk")
        if use_live and collapse_model is not None and collapse_cal is not None:
            runs_n = max(0, score_1st + 1 - score_2nd)
            balls_left = max(1, (20 - current_over) * 6)
            out = live_collapse_probability(runs_needed=runs_n, balls_remaining=balls_left, wickets_fallen=wkts_2nd, batter_sr_recent=float(batter_sr), partnership_balls=partnership, bowler_economy=bowler_eco, model=collapse_model, calibrator=collapse_cal)
            risk, prob = out.get("risk_tier", "low"), out.get("collapse_probability", 0) * 100
            st.metric("Risk level", risk.upper(), f"{prob:.0f}% in next 30 balls")
            if risk in ("critical", "high"):
                st.warning("Elevated collapse risk.")
        else:
            st.info("Set live score in the sidebar to see collapse risk during the chase.")

with col2:
    with st.container():
        st.subheader("Toss at venue")
        if toss_df is not None:
            try:
                mod = toss_df[(toss_df["venue"] == venue) & (toss_df["era"] == ERA_FUTURE)]
                if not mod.empty:
                    bat_row = mod[mod["toss_decision"] == "bat"]
                    fld_row = mod[mod["toss_decision"] == "field"]
                    bat_e = bat_row["toss_edge"].iloc[0] if len(bat_row) else 0
                    fld_e = fld_row["toss_edge"].iloc[0] if len(fld_row) else 0
                    st.caption(f"Bat first: {bat_e:+.2f} · Field first: {fld_e:+.2f}")
                    if abs(toss_edge_score) < 0.02:
                        st.success("Toss impact at this venue is roughly neutral.")
                    else:
                        st.success(f"Toss favours **{team_a if toss_edge_score > 0 else team_b}** here.")
                else:
                    st.caption("No venue data for modern era.")
            except Exception:
                st.caption("Toss data unavailable.")
        else:
            st.caption("Run pipeline to load toss data.")

with st.expander("How to use"):
    st.markdown("**Pre-match:** Leave over and 2nd innings score at 0. **Live:** Enter current over, 1st innings total, and 2nd innings score to update win probability and collapse risk.")
