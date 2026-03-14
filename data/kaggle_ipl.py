"""
Download and load IPL ball-by-ball data from Kaggle.
Dataset: https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025/data

Requires Kaggle API credentials:
  1. Kaggle account -> Account -> Create New API Token (downloads kaggle.json)
  2. Put kaggle.json in one of:
     - ~/.kaggle/kaggle.json
     - Desktop/kaggle/kaggle.json (this project will use it automatically)
     - Or any folder and set env: export KAGGLE_CONFIG_DIR=/path/to/folder
  3. pip install kaggle
"""

import os
from pathlib import Path

import pandas as pd

# Use Desktop/kaggle if it exists and default ~/.kaggle has no key
if "KAGGLE_CONFIG_DIR" not in os.environ:
    for candidate in [
        Path.home() / "Desktop" / "kaggle",
        Path.home() / "Desktop" / "Kaggle",
    ]:
        if (candidate / "kaggle.json").exists():
            os.environ["KAGGLE_CONFIG_DIR"] = str(candidate)
            break

# Dataset identifier on Kaggle
KAGGLE_DATASET = "chaitu20/ipl-dataset2008-2025"

# Map common Kaggle IPL column names -> our schema (loader.REQUIRED_COLS)
# Update this if the actual CSV uses different names
KAGGLE_COLUMN_MAP = {
    "id": "match_id",
    "match_id": "match_id",
    "match_number": "match_id",
    "inning": "innings",
    "innings": "innings",
    "batting_team": "batting_team",
    "bowling_team": "bowling_team",
    "over": "over",
    "ball": "ball_no",
    "ball_no": "ball_no",
    "batsman": "batter",
    "batter": "batter",
    "batsman_runs": "runs_batter",
    "runs_batter": "runs_batter",
    "total_runs": "runs_total",
    "runs_total": "runs_total",
    "extra_runs": "extra_runs",  # keep to derive valid_ball
    "dismissal_kind": "wicket_kind",
    "wicket_kind": "wicket_kind",
    "player_dismissed": "player_out",
    "player_out": "player_out",
    "toss_winner": "toss_winner",
    "toss_decision": "toss_decision",
    "winner": "match_won_by",
    "match_won_by": "match_won_by",
    "venue": "venue",
    "date": "date",
    "season": "season",
    "city": "venue",  # fallback if venue missing
}


def download_ipl_kaggle(dest_dir: str = "data/kaggle_ipl") -> str:
    """
    Download the IPL dataset from Kaggle and unzip into dest_dir.
    Returns the path to the directory containing CSV files.

    Requires: pip install kaggle, and ~/.kaggle/kaggle.json configured.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Kaggle API required. Install with: pip install kaggle\n"
            "Then set up API key: https://www.kaggle.com/settings -> Create New Token, save to ~/.kaggle/kaggle.json"
        )

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(dest), unzip=True)
    print(f"Downloaded to {dest.absolute()}")
    return str(dest)


def _find_ball_by_ball_csv(data_dir: str) -> str:
    """Find the main ball-by-ball CSV in the downloaded folder."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Common filenames for ball-by-ball data
    candidates = list(data_path.glob("**/*.csv"))
    # Prefer files that look like ball-by-ball (larger, or name contains ball/delivery)
    def score(p: Path) -> int:
        s = p.name.lower()
        n = p.stat().st_size
        if "ball" in s or "delivery" in s or "deliveries" in s:
            return 1000 + n
        if "match" in s and "ball" not in s:
            return -1000  # often match summary
        return n

    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    best = max(candidates, key=score)
    return str(best)


def _rename_and_derive(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to our schema and derive missing required columns."""
    # Apply renames (only columns that exist) while avoiding duplicate target columns.
    rename = {}
    for k, v in KAGGLE_COLUMN_MAP.items():
        if k in df.columns and k != v:
            if v in df.columns and v != k:
                # Prefer the existing target column; skip duplicates.
                continue
            rename[k] = v
    df = df.rename(columns=rename)

    # Derive ball_no from over + ball if present (avoid floating ball_no like 0.1).
    if "over" in df.columns and "ball" in df.columns:
        df["over"] = pd.to_numeric(df["over"], errors="coerce").fillna(0).astype(int)
        df["ball"] = pd.to_numeric(df["ball"], errors="coerce").fillna(0).astype(int)
        df["ball_no"] = df["over"] * 6 + df["ball"]
        df = df.drop(columns=["ball"], errors="ignore")

    # Ensure match_id is string for consistency
    if "match_id" in df.columns:
        df["match_id"] = df["match_id"].astype(str)

    # date / year / season
    if "date" not in df.columns and "season" in df.columns:
        df["date"] = pd.to_datetime(df["season"].astype(str) + "-01-01", errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "season" not in df.columns and "date" in df.columns:
        df["season"] = pd.to_datetime(df["date"]).dt.year

    # Normalize season to integer year (handles strings like 2007/08)
    if "season" in df.columns:
        season_year = pd.to_numeric(
            df["season"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
        )
        season_year = season_year.fillna(pd.to_datetime(df["date"], errors="coerce").dt.year)
        df["season"] = season_year.fillna(0).astype(int)

    # valid_ball: 1 for normal deliveries; 0 for wides/noballs if we have extra_runs
    if "valid_ball" not in df.columns:
        if "extra_runs" in df.columns:
            # Consider only runs from bat as valid ball (simplified: 0 if extra_runs > 0 and no runs_batter?)
            df["valid_ball"] = 1
            # Optional: set 0 for wide/noby so they don't count as balls for pressure
            # df.loc[df["extra_runs"].fillna(0) > 0, "valid_ball"] = 0  # uncomment if needed
        else:
            df["valid_ball"] = 1

    # runs_target: for 2nd innings, target = 1st innings total + 1
    if "runs_target" not in df.columns and "innings" in df.columns:
        first_innings_total = (
            df[df["innings"] == 1]
            .groupby("match_id")["runs_total"]
            .sum()
            .reset_index()
            .rename(columns={"runs_total": "first_innings_runs"})
        )
        df = df.merge(first_innings_total, on="match_id", how="left")
        df["runs_target"] = df.loc[df["innings"] == 2, "first_innings_runs"] + 1
        df = df.drop(columns=["first_innings_runs"], errors="ignore")

    # wicket_kind: NaN when no wicket
    if "wicket_kind" not in df.columns:
        df["wicket_kind"] = pd.NA

    # toss_winner / toss_decision / match_won_by: sometimes in a separate matches file
    # If missing, we need to merge from a match-level file in the same folder
    for col in ["toss_winner", "toss_decision", "match_won_by", "venue"]:
        if col not in df.columns:
            df[col] = None if col != "venue" else ""

    return df


def _add_event_match_no(df: pd.DataFrame) -> pd.DataFrame:
    """Derive event_match_no from match_id order (by date) and overwrite old values.
    Required for pre_match_features."""
    matches = df.drop_duplicates("match_id")[["match_id", "date", "season"]].copy()
    matches = matches.sort_values(["season", "date", "match_id"]).reset_index(drop=True)
    matches["event_match_no"] = range(1, len(matches) + 1)
    # Always regenerate to ensure chronological ordering and numeric type.
    df = df.drop(columns=["event_match_no"], errors="ignore")
    df = df.merge(matches[["match_id", "event_match_no"]], on="match_id", how="left")
    return df


def load_ipl_from_kaggle(
    data_dir: str = "data/kaggle_ipl",
    *,
    download_if_missing: bool = True,
    column_map: dict = None,
) -> pd.DataFrame:
    """
    Load IPL data from a Kaggle-downloaded folder (or download first if missing).

    If download_if_missing is True and data_dir is empty/missing, downloads
    the dataset from Kaggle (requires kaggle.json in ~/.kaggle/).

    Returns a DataFrame in the same schema as load_ipl() for use with the rest of the pipeline.
    """
    data_dir = Path(data_dir)
    if download_if_missing and (not data_dir.exists() or not list(data_dir.glob("**/*.csv"))):
        data_dir = Path(download_ipl_kaggle(str(data_dir)))

    csv_path = _find_ball_by_ball_csv(str(data_dir))
    df = pd.read_csv(csv_path, low_memory=False)

    if column_map:
        global KAGGLE_COLUMN_MAP
        orig = KAGGLE_COLUMN_MAP.copy()
        KAGGLE_COLUMN_MAP.update(column_map)
    try:
        df = _rename_and_derive(df)
    finally:
        if column_map:
            KAGGLE_COLUMN_MAP.clear()
            KAGGLE_COLUMN_MAP.update(orig)

    # Merge match-level info from a separate "matches" CSV if present (toss, winner, venue)
    data_path = Path(data_dir)
    for p in data_path.rglob("*.csv"):
        if Path(csv_path).resolve() == p.resolve():
            continue
        name = p.stem.lower()
        if "match" in name and "ball" not in name and "delivery" not in name:
            try:
                match_df = pd.read_csv(p, low_memory=False)
                mid_col = "id" if "id" in match_df.columns else "match_id"
                if mid_col not in match_df.columns:
                    continue
                match_df["match_id"] = match_df[mid_col].astype(str)
                if "winner" in match_df.columns:
                    match_df["match_won_by"] = match_df["winner"]
                merge_cols = ["match_id"]
                for c in ["toss_winner", "toss_decision", "match_won_by", "venue", "date", "season"]:
                    if c in match_df.columns:
                        merge_cols.append(c)
                if len(merge_cols) > 1:
                    to_drop = [c for c in merge_cols if c in df.columns and c != "match_id"]
                    df = df.drop(columns=to_drop, errors="ignore")
                    df = df.merge(match_df[merge_cols].drop_duplicates("match_id"), on="match_id", how="left")
            except Exception as e:
                print(f"Note: could not merge match-level file {p}: {e}")
            break

    df = _add_event_match_no(df)

    # Final step: run the same cleaning as loader.load_ipl
    from data.loader import _clean_ipl_dataframe
    df = _clean_ipl_dataframe(df)
    return df


def get_data_path() -> str:
    """Return default path for Kaggle IPL data (for use in notebooks/scripts)."""
    return "data/kaggle_ipl"
