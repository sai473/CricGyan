"""
IPL Match Predictor
Run: python3 -m streamlit run dashboard/app.py
"""

import streamlit as st
import pickle
import os
import sys
import io
import urllib.request
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.pressure_index import live_collapse_probability
from models.unified_predictor import predict_match
from models.toss_alpha_decay import get_toss_edge_feature


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_models_from_remote():
    """If models are missing, optionally fetch a zip from st.secrets MODEL_BUNDLE_URL (Streamlit Cloud)."""
    saved = os.path.join(_project_root(), "models", "saved")
    if os.path.isfile(os.path.join(saved, "lgb_pre.pkl")):
        return
    try:
        url = st.secrets["MODEL_BUNDLE_URL"]
    except KeyError:
        return
    os.makedirs(saved, exist_ok=True)
    try:
        with st.spinner("Downloading model bundle…"):
            req = urllib.request.Request(url, headers={"User-Agent": "ipl-intelligence-engine/1"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(saved)
    except Exception as e:
        st.error(f"Could not download or unpack model bundle: {e}")


st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏", layout="centered", initial_sidebar_state="expanded")
_ensure_models_from_remote()

# Narrow, readable column; rely on Streamlit defaults otherwise
st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; max-width: 42rem; }
    #MainMenu, footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


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
    st.error(
        "Models not found locally. Either run `python3 run_from_kaggle.py`, "
        "commit `models/saved/*.pkl`, or set **MODEL_BUNDLE_URL** in Streamlit Cloud secrets "
        "(see `.streamlit/secrets.toml.example`)."
    )
    st.stop()

TEAMS = [
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore", "Kolkata Knight Riders",
    "Rajasthan Royals", "Delhi Capitals", "Sunrisers Hyderabad", "Punjab Kings", "Gujarat Titans", "Lucknow Super Giants",
]

# --- Sidebar ---
with st.sidebar:
    st.subheader("Match")
    team_a = st.selectbox("Batting first", TEAMS, key="team_a")
    team_b = st.selectbox("Chasing", TEAMS, key="team_b")
    venue_options = [
        "Wankhede Stadium", "MA Chidambaram Stadium", "Eden Gardens", "M Chinnaswamy Stadium", "Arun Jaitley Stadium",
        "Rajiv Gandhi International Stadium", "Sawai Mansingh Stadium", "Narendra Modi Stadium", "Dubai International Cricket Stadium",
        "Sharjah Cricket Stadium", "Sheikh Zayed Stadium", "Brabourne Stadium",
    ]
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
    st.caption("Set over and scores for in-play odds.")
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
fav = team_a if win_a >= 50 else team_b

# --- Main ---
st.title("IPL Match Predictor")
if use_live:
    st.caption("In-play · win and collapse use live state from the sidebar")
else:
    st.caption("Pre-match · set over to 0 and 2nd innings score to 0 for static odds")

st.divider()

a_col, b_col = st.columns(2)
with a_col:
    st.metric(team_a, f"{win_a:.1f}%", help="Team batting first")
with b_col:
    st.metric(team_b, f"{win_b:.1f}%", help="Chasing team")

st.write("")
st.caption(f"{team_a} · {win_a:.0f}%")
st.progress(win_a / 100.0)
st.caption(f"{team_b} · {win_b:.0f}%")
st.progress(win_b / 100.0)

st.caption(f"Favourite: **{fav}** · Toss edge (batting first): {toss_edge_score:+.3f}")

st.divider()
st.subheader("Collapse risk (next 30 balls)")
if use_live and collapse_model is not None and collapse_cal is not None:
    runs_n = max(0, score_1st + 1 - score_2nd)
    balls_left = max(1, (20 - current_over) * 6)
    out = live_collapse_probability(
        runs_needed=runs_n, balls_remaining=balls_left, wickets_fallen=wkts_2nd,
        batter_sr_recent=float(batter_sr), partnership_balls=partnership, bowler_economy=bowler_eco,
        model=collapse_model, calibrator=collapse_cal,
    )
    risk = (out.get("risk_tier") or "low").lower()
    prob = out.get("collapse_probability", 0) * 100
    st.metric("Estimated chance (30 balls)", f"{prob:.0f}%", delta=f"Tier: {risk}")
    if risk in ("critical", "high"):
        st.warning("Higher chance of wickets falling in a cluster soon.")
else:
    st.info("Turn on **live** mode: set **Over** above 0 and **1st innings total** in the sidebar.")

st.divider()
st.subheader("Toss at venue")
st.caption(venue)
if toss_df is not None:
    try:
        mod = toss_df[(toss_df["venue"] == venue) & (toss_df["era"] == ERA_FUTURE)]
        if not mod.empty:
            bat_row = mod[mod["toss_decision"] == "bat"]
            fld_row = mod[mod["toss_decision"] == "field"]
            bat_e = bat_row["toss_edge"].iloc[0] if len(bat_row) else 0
            fld_e = fld_row["toss_edge"].iloc[0] if len(fld_row) else 0
            t1, t2 = st.columns(2)
            t1.metric("Bat first", f"{bat_e:+.2f}")
            t2.metric("Field first", f"{fld_e:+.2f}")
            if abs(toss_edge_score) < 0.02:
                st.success("Toss effect at this venue looks small.")
            else:
                fav_team = team_a if toss_edge_score > 0 else team_b
                st.success(f"Model leans toward **{fav_team}** after the toss here.")
        else:
            st.caption("No row for this venue in the modern-era table.")
    except Exception:
        st.caption("Could not read toss table.")
else:
    st.caption("Train the pipeline to load toss data.")

with st.expander("How to use"):
    st.markdown(
        "- **Pre-match:** Over = 0, 2nd innings score = 0.\n"
        "- **Live:** Set over, 1st-innings total, and current score.\n"
        "- Change **venue** and **toss** anytime; numbers update immediately."
    )
