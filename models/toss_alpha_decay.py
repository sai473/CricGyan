"""
Module 2: Toss Alpha Decay
==========================
Computes how much the toss advantage has eroded across 18 IPL seasons,
per venue and toss decision (bat / field first).

Uses Bayesian Beta posteriors so small-sample venues get shrunk toward 0.5.
"""

import pandas as pd
import numpy as np
from scipy.stats import beta as beta_dist


ERA_BINS   = [2007, 2012, 2018, 2025]
ERA_LABELS = ['early', 'mid', 'modern']


# ── Core computation ──────────────────────────────────────────────────────────

def compute_toss_alpha_decay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a lookup table: venue × toss_decision × era
    with columns: win_rate_bayes, toss_edge, alpha_decay_rate.
    """
    matches = df.drop_duplicates('match_id').copy()
    matches['era'] = pd.cut(
        matches['year'].astype(float), bins=ERA_BINS, labels=ERA_LABELS
    )
    matches['toss_winner_won'] = (
        matches['toss_winner'] == matches['match_won_by']
    ).astype(int)

    agg = (
        matches
        .groupby(['venue', 'toss_decision', 'era'], observed=True)
        .agg(matches=('match_id', 'count'), wins=('toss_winner_won', 'sum'))
        .reset_index()
    )

    # Beta posterior (uniform prior: α=1, β=1)
    agg['α'] = agg['wins'] + 1
    agg['β'] = (agg['matches'] - agg['wins']) + 1
    agg['win_rate_bayes'] = agg['α'] / (agg['α'] + agg['β'])
    agg['win_rate_lower'] = beta_dist.ppf(0.10, agg['α'], agg['β'])
    agg['win_rate_upper'] = beta_dist.ppf(0.90, agg['α'], agg['β'])
    agg['toss_edge']      = agg['win_rate_bayes'] - 0.5

    # Alpha decay: how much edge shrank from early → modern
    early  = agg[agg['era'] == 'early' ][['venue','toss_decision','toss_edge']].rename(columns={'toss_edge':'edge_early'})
    modern = agg[agg['era'] == 'modern'][['venue','toss_decision','toss_edge']].rename(columns={'toss_edge':'edge_modern'})
    decay  = early.merge(modern, on=['venue','toss_decision'], how='inner')
    decay['alpha_decay_rate'] = (
        (decay['edge_early'] - decay['edge_modern']) /
        (decay['edge_early'].abs() + 1e-6)
    ).clip(-2, 2)

    result = agg.merge(
        decay[['venue','toss_decision','alpha_decay_rate']],
        on=['venue','toss_decision'], how='left'
    )
    return result


def get_toss_edge_feature(
    toss_alpha_df: pd.DataFrame,
    venue: str,
    toss_decision: str,
    era: str,
) -> float:
    """
    Lookup the Bayesian toss edge for a specific match context.
    Returns 0.0 (neutral) if venue is unseen.
    """
    row = toss_alpha_df[
        (toss_alpha_df['venue'] == venue) &
        (toss_alpha_df['toss_decision'] == toss_decision) &
        (toss_alpha_df['era'] == era)
    ]
    return float(row['toss_edge'].iloc[0]) if not row.empty else 0.0


# ── Summary & reporting ───────────────────────────────────────────────────────

def print_decay_report(toss_alpha_df: pd.DataFrame) -> None:
    """Print a human-readable decay summary for every venue."""
    print("\n  TOSS ALPHA DECAY REPORT")
    print("  " + "─" * 65)
    print(f"  {'Venue':<30} {'Decision':<8} {'Early':>7} {'Mid':>7} {'Modern':>8} {'Decay%':>8}")
    print("  " + "─" * 65)

    venues = toss_alpha_df['venue'].unique()
    for venue in sorted(venues):
        for dec in ['bat', 'field']:
            rows = toss_alpha_df[
                (toss_alpha_df['venue'] == venue) &
                (toss_alpha_df['toss_decision'] == dec)
            ].set_index('era')['toss_edge']
            e = rows.get('early',  0.0)
            m = rows.get('mid',    0.0)
            mod = rows.get('modern', 0.0)
            decay_pct = (e - mod) / (abs(e) + 1e-6) * 100
            flag = ' ← eroded' if abs(e) > 0.04 and decay_pct > 50 else ''
            print(f"  {venue[:30]:<30} {dec:<8} {e:+.3f}  {m:+.3f}   {mod:+.3f}   {decay_pct:+.0f}%{flag}")
    print()


def significant_venues(toss_alpha_df: pd.DataFrame, era: str = 'modern', min_edge: float = 0.04) -> pd.DataFrame:
    """Return venues where the toss still confers a meaningful edge in the given era."""
    modern = toss_alpha_df[toss_alpha_df['era'] == era].copy()
    return (
        modern[modern['toss_edge'].abs() >= min_edge]
        .sort_values('toss_edge', ascending=False)
        [['venue','toss_decision','win_rate_bayes','toss_edge','win_rate_lower','win_rate_upper','matches']]
    )
