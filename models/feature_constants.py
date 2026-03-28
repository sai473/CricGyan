"""
Feature name lists for inference (no heavy imports). Kept in sync with training modules.
"""

PRE_MATCH_FEATURES = [
    "toss_edge_score",
    "elo_delta",
    "form_delta",
    "h2h_venue_wr",
    "is_playoff",
    "match_number_norm",
]

# In-match / chase snapshot features (same as COLLAPSE_FEATURES in pressure_index.py)
IN_MATCH_FEATURES = [
    "rrr",
    "crr",
    "rrr_gap",
    "rrr_ratio",
    "wickets_fallen",
    "wickets_remaining",
    "phase_pct",
    "is_powerplay",
    "is_death",
    "batter_sr_recent",
    "partnership_ball",
    "bowler_economy_live",
    "pressure_index",
    "runs_needed",
    "balls_remaining",
]

COLLAPSE_FEATURES = IN_MATCH_FEATURES
