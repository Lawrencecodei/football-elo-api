"""
elo_model.py
------------
Standalone Elo implementation plus a helper to update ratings
from a DataFrame of historical matches.

Author: you :)
"""

from typing import Dict, Tuple
import pandas as pd

# ---------- Hyper-parameters ----------
K_DEFAULT = 20        # learning rate (bigger = react faster)
HOME_ADVANTAGE = 65   # Elo points given to the home team


# ---------- Core Elo class ----------
class EloRating:
    """Lightweight Elo tracker for football teams."""

    def __init__(self, base_rating: int = 1500, k: int = K_DEFAULT):
        self.ratings: Dict[str, float] = {}
        self.base_rating = base_rating
        self.k = k

    # ---- helpers ----------------------------------------------------------

    def get_rating(self, team: str) -> float:
        """Current rating, or the base if team is unseen."""
        return self.ratings.get(team, self.base_rating)

    @staticmethod
    def expected(ra: float, rb: float) -> float:
        """Win probability for team-A given ratings ra, rb."""
        return 1 / (1 + 10 ** ((rb - ra) / 400))

    # ---- update -----------------------------------------------------------

    def update(
        self,
        home: str,
        away: str,
        result: float,  # 1.0 = home win, 0.5 = draw, 0.0 = away win
    ) -> Tuple[float, float, float]:
        """
        Adjust ratings after a match and return:
        (new_home_rating, new_away_rating, home_expected_prob)
        """
        # Add home-field advantage only for expectancy calc
        ra_adj = self.get_rating(home) + HOME_ADVANTAGE
        rb = self.get_rating(away)
        ea = self.expected(ra_adj, rb)
        eb = 1.0 - ea

        # Update ratings (without HFA)
        new_home = self.get_rating(home) + self.k * (result - ea)
        new_away = self.get_rating(away) + self.k * ((1 - result) - eb)
        self.ratings[home] = new_home
        self.ratings[away] = new_away
        return new_home, new_away, ea


# ---------- Batch helper ---------------------------------------------------
def update_ratings_from_matches(
    df: pd.DataFrame, elo: EloRating
) -> pd.DataFrame:
    """
    Walk a DataFrame chronologically and update Elo each match.
    Expects columns:
        date, home_team, away_team, full_time_result   (H / D / A)
    Returns a history DataFrame with pre/post ratings & expected win prob.
    """
    res_map = {"H": 1.0, "D": 0.5, "A": 0.0}
    history = []

    for _, row in df.sort_values("date").iterrows():
        result = res_map[row["full_time_result"]]
        pre_home = elo.get_rating(row["home_team"])
        pre_away = elo.get_rating(row["away_team"])
        new_home, new_away, exp_home = elo.update(
            row["home_team"], row["away_team"], result
        )
        history.append(
            {
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "result": row["full_time_result"],
                "exp_home_win": round(exp_home, 3),
                "home_elo_pre": pre_home,
                "away_elo_pre": pre_away,
                "home_elo_post": new_home,
                "away_elo_post": new_away,
            }
        )

    return pd.DataFrame(history)
