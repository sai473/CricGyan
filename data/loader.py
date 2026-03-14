import pandas as pd
import numpy as np


ERA_BINS   = [2007, 2012, 2018, 2025]
ERA_LABELS = ['early', 'mid', 'modern']

REQUIRED_COLS = [
    'match_id', 'innings', 'batting_team', 'bowling_team',
    'over', 'ball_no', 'batter', 'runs_batter', 'runs_total',
    'wicket_kind', 'runs_target', 'toss_winner', 'toss_decision',
    'match_won_by', 'venue', 'season', 'valid_ball',
]


def _clean_ipl_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply baseline cleaning to a DataFrame that already has required column names."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    if 'date' not in df.columns:
        if 'season' in df.columns:
            df['date'] = pd.to_datetime(df['season'].astype(str) + '-01-01', errors='coerce')
        else:
            df['date'] = pd.NaT
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year.fillna(df.get('year', pd.Series(dtype=int))).astype('Int64')
    df['era'] = pd.cut(df['year'].astype(float), bins=ERA_BINS, labels=ERA_LABELS)

    df['over'] = pd.to_numeric(df['over'], errors='coerce').fillna(0).astype(int)
    df['ball_no'] = pd.to_numeric(df['ball_no'], errors='coerce').fillna(0).astype(int)
    df['runs_batter'] = pd.to_numeric(df['runs_batter'], errors='coerce').fillna(0)
    df['runs_total'] = pd.to_numeric(df['runs_total'], errors='coerce').fillna(0)
    df['valid_ball'] = pd.to_numeric(df['valid_ball'], errors='coerce').fillna(1).astype(int)

    df = df.sort_values(['match_id', 'innings', 'ball_no']).reset_index(drop=True)
    print(f"Loaded {len(df):,} balls across {df['match_id'].nunique():,} matches "
          f"({df['season'].nunique()} seasons).")
    return df


def load_ipl(path: str) -> pd.DataFrame:
    """Load raw IPL CSV and apply baseline cleaning."""
    df = pd.read_csv(path, low_memory=False)
    return _clean_ipl_dataframe(df)


def get_match_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse ball-by-ball to one row per match with key match facts."""
    return (
        df.drop_duplicates('match_id')
          .assign(year=lambda d: d['year'].astype('Int64'))
          .copy()
    )


def split_by_season(df: pd.DataFrame, train_end: int = 2022, val_year: int = 2023):
    """Chronological train / val / test split — never random."""
    train = df[df['season'] <= train_end]
    val   = df[df['season'] == val_year]
    test  = df[df['season'] >  val_year]
    print(f"Train: {train['match_id'].nunique()} matches | "
          f"Val: {val['match_id'].nunique()} | "
          f"Test: {test['match_id'].nunique()}")
    return train, val, test
