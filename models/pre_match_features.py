"""
Module 3: Pre-Match Feature Engineering
========================================
Builds one row per match with all signals knowable BEFORE ball 1:
  - ELO-style rolling team strength
  - Rolling 5-match win rate (form)
  - Head-to-head at this specific venue
  - Toss edge score (from toss_alpha_decay module)
  - Match context (stage, season phase)
"""

import pandas as pd
import numpy as np
from models.toss_alpha_decay import compute_toss_alpha_decay, get_toss_edge_feature


ERA_BINS   = [2007, 2012, 2018, 2025]
ERA_LABELS = ['early', 'mid', 'modern']

PRE_MATCH_FEATURES = [
    'toss_edge_score',
    'elo_delta',
    'form_delta',
    'h2h_venue_wr',
    'is_playoff',
    'match_number_norm',
]


# ── ELO ───────────────────────────────────────────────────────────────────────

def compute_elo(matches: pd.DataFrame, K: int = 32) -> pd.DataFrame:
    """
    Compute ELO rating for every team at the START of each match
    (i.e., no leakage — we use the rating before the match is played).
    """
    matches = matches.sort_values(['year', 'event_match_no']).copy()
    elo = {t: 1500.0 for t in pd.concat([matches['batting_team'], matches['bowling_team']]).unique()}
    records = []

    for _, row in matches.iterrows():
        tA, tB = row['batting_team'], row['bowling_team']
        records.append({
            'match_id':    row['match_id'],
            'elo_team_a':  elo.get(tA, 1500),
            'elo_team_b':  elo.get(tB, 1500),
        })
        # Update after recording pre-match values
        winner = row.get('match_won_by')
        if winner in (tA, tB):
            w, l = (tA, tB) if winner == tA else (tB, tA)
            ea = 1 / (1 + 10 ** ((elo[l] - elo[w]) / 400))
            elo[w] += K * (1 - ea)
            elo[l] += K * (0 - (1 - ea))

    elo_df = pd.DataFrame(records)
    matches = matches.merge(elo_df, on='match_id', how='left')
    matches['elo_delta'] = matches['elo_team_a'] - matches['elo_team_b']
    return matches


# ── Rolling form ──────────────────────────────────────────────────────────────

def add_rolling_form(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    matches = matches.sort_values(['year', 'event_match_no']).copy()

    def team_form(team_col):
        won = (matches[team_col] == matches['match_won_by']).astype(int)
        return (
            matches.assign(_won=won)
                   .groupby(team_col)['_won']
                   .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    matches['form_a']    = team_form('batting_team')
    matches['form_b']    = team_form('bowling_team')
    matches['form_delta'] = matches['form_a'].fillna(0.5) - matches['form_b'].fillna(0.5)
    return matches


# ── Head-to-head at venue ─────────────────────────────────────────────────────

def add_h2h_venue(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.sort_values(['year', 'event_match_no']).copy()
    h2h_wr = []

    for _, row in matches.iterrows():
        tA, tB, v = row['batting_team'], row['bowling_team'], row['venue']
        past = matches[
            (matches['event_match_no'] < row['event_match_no']) &
            (matches['venue'] == v) &
            (
                ((matches['batting_team'] == tA) & (matches['bowling_team'] == tB)) |
                ((matches['batting_team'] == tB) & (matches['bowling_team'] == tA))
            )
        ]
        if past.empty:
            h2h_wr.append(0.5)
        else:
            h2h_wr.append((past['match_won_by'] == tA).sum() / len(past))

    matches['h2h_venue_wr'] = h2h_wr
    return matches


# ── Main builder ──────────────────────────────────────────────────────────────

def build_pre_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pre-match feature pipeline. Call this once on your full dataset.
    Returns one row per match with PRE_MATCH_FEATURES + 'team_a_won' label.
    """
    matches = df.drop_duplicates('match_id').copy()
    matches['era'] = pd.cut(
        matches['year'].astype(float), bins=ERA_BINS, labels=ERA_LABELS
    )

    # Build toss decay table from the same data (chronological — no leakage
    # because toss_edge is an aggregate over all historical matches, not
    # the current one; use only past seasons in production)
    toss_alpha_df = compute_toss_alpha_decay(df)

    matches = compute_elo(matches)
    matches = add_rolling_form(matches)
    matches = add_h2h_venue(matches)

    # Toss edge: positive = team_a (batting_team) benefited from toss
    matches['toss_edge_score'] = matches.apply(
        lambda r: get_toss_edge_feature(
            toss_alpha_df,
            r['venue'],
            r.get('toss_decision', 'bat'),
            str(r['era']),
        ) * (1 if r.get('toss_winner') == r['batting_team'] else -1),
        axis=1,
    )

    matches['is_playoff'] = (
        matches.get('stage', pd.Series('', index=matches.index))
               .str.contains('Final|Semi|Qualifier|Eliminator', na=False)
    ).astype(int)

    matches['match_number_norm'] = (
        matches['event_match_no'] / matches['event_match_no'].max()
    )

    matches['team_a_won'] = (
        matches['match_won_by'] == matches['batting_team']
    ).astype(int)

    keep = ['match_id', 'year', 'season', 'venue', 'era',
            'batting_team', 'bowling_team',
            'elo_team_a', 'elo_team_b', 'form_a', 'form_b',
            ] + PRE_MATCH_FEATURES + ['team_a_won']
    return matches[[c for c in keep if c in matches.columns]]
