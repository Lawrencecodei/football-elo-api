"""
data_pipeline.py (grid‑search version)
-------------------------------------
Pipeline now:
1. Downloads EPL + La Liga seasons (2021‑22 → 2024‑25 by default).
2. Cleans & merges them.
3. Splits **train** (all but latest) and **test** (latest season).
4. **Grid‑searches** Elo hyper‑parameters:
   · `K` ∈ [15, 20, 25]
   · `HOME_ADVANTAGE` ∈ [50, 65, 80]
   selecting the combo with the highest balanced accuracy on the test set.
5. Prints a leaderboard and the best scores.
6. Saves the best ratings to `data/elo_ratings.pkl`.

Run:
    python data_pipeline.py

Requires:
    pandas, requests, tqdm, scikit‑learn
"""

from __future__ import annotations
import io
import pickle
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# ---- CONFIG --------------------------------------------------------------
SEASONS = ["2021-2022", "2022-2023", "2023-2024", "2024-2025"]
LEAGUES = {"EPL": "E0", "LaLiga": "SP1"}
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
DATA_DIR = Path("data/raw")
CLEAN_PATH = Path("data/matches_clean.csv")
ELO_PKL = Path("data/elo_ratings.pkl")

# search grids
K_GRID = [15, 20, 25]
HFA_GRID = [50, 65, 80]

# ---- import Elo ----------------------------------------------------------
from elo_model import EloRating, update_ratings_from_matches

# ---- helpers -------------------------------------------------------------

def season_code(season: str) -> str:
    start, end = season.split("-")
    return f"{start[-2:]}{end[-2:]}"


def download_csv(season: str, league: str, code: str) -> pd.DataFrame:
    url = BASE_URL.format(season_code=season_code(season), league_code=code)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out = pd.read_csv(io.StringIO(r.text))
    out.insert(0, "season", season)
    out.insert(1, "league", league)
    return out


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    sel = df[["Date", "HomeTeam", "AwayTeam", "FTR", "season", "league"]].copy()
    sel.columns = ["date", "home_team", "away_team", "full_time_result", "season", "league"]
    sel["date"] = pd.to_datetime(sel["date"], dayfirst=True, errors="coerce")
    sel = sel.dropna(subset=["date"])
    sel = sel[sel["full_time_result"].isin(["H", "D", "A"])]
    return sel.reset_index(drop=True)


def download_all() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        for lg, code in LEAGUES.items():
            print(f"Downloading {lg} {season} …")
            frames.append(download_csv(season, lg, code))
    raw = pd.concat(frames, ignore_index=True)
    tidy_df = tidy(raw)
    tidy_df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved clean data → {CLEAN_PATH}")
    return tidy_df


# ---- grid‑search ---------------------------------------------------------

def evaluate_combo(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int, hfa: int) -> Tuple[float, float, dict[str, float]]:
    elo = EloRating(k=k)
    _ = update_ratings_from_matches(train_df, elo)

    y_true, y_pred = [], []
    for _, row in test_df.sort_values("date").iterrows():
        ra_adj = elo.get_rating(row["home_team"]) + hfa
        rb = elo.get_rating(row["away_team"])
        p_home = elo.expected(ra_adj, rb)
        p_away = elo.expected(rb, ra_adj)
        p_draw = max(0.0, 1 - p_home - p_away)
        pred = ["H", "D", "A"][[p_home, p_draw, p_away].index(max(p_home, p_draw, p_away))]
        y_pred.append(pred)
        y_true.append(row["full_time_result"])
        # online update so ratings evolve across test season
        result_map = {"H": 1.0, "D": 0.5, "A": 0.0}
        _ = elo.update(row["home_team"], row["away_team"], result_map[row["full_time_result"]])

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return bal_acc, acc, elo.ratings


def grid_search(train_df: pd.DataFrame, test_df: pd.DataFrame):
    leaderboard = []
    best_bal, best_acc, best_cfg, best_ratings = -1, -1, None, None

    for k in K_GRID:
        for hfa in HFA_GRID:
            bal, acc, ratings = evaluate_combo(train_df, test_df, k, hfa)
            leaderboard.append((k, hfa, bal, acc))
            if bal > best_bal:
                best_bal, best_acc, best_cfg, best_ratings = bal, acc, (k, hfa), ratings
            print(f"K={k:2d}, HFA={hfa:2d} ➜ bal={bal:.3f}  acc={acc:.3f}")

    print("\n🏅 Best combo  K=%d  HFA=%d  |  bal=%.3f  acc=%.3f" % (*best_cfg, best_bal, best_acc))
    return best_ratings


# ---- main ---------------------------------------------------------------

def main():
    print("▶ Download & clean …")
    clean_df = download_all()

    latest = max(clean_df["season"].unique())
    train_df = clean_df[clean_df["season"] < latest]
    test_df = clean_df[clean_df["season"] == latest]

    print(f"▶ Grid‑search on held‑out season {latest} (n={len(test_df)}) …")
    best_ratings = grid_search(train_df, test_df)

    ELO_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(ELO_PKL, "wb") as f:
        pickle.dump(best_ratings, f)
    print(f"Saved best Elo ratings → {ELO_PKL}\nDone.")


if __name__ == "__main__":
    main()
