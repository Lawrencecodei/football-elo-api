# elo_fastapi_app.py
# ------------------
# FastAPI wrapper that loads pre-computed Elo ratings
# and serves predictions at /predict.

from fastapi import FastAPI
from pydantic import BaseModel

# 1) 🔗  NEW IMPORT — pulls the Elo code from elo_model.py
from elo_model import EloRating, update_ratings_from_matches
# 2) utils to load the pickle
import pickle, pathlib

# ---------- initialize app ----------
app = FastAPI(title="Football Elo Predictor", version="0.2.0")

# ---------- load Elo ratings ----------
ELO_PATH = pathlib.Path("data/elo_ratings.pkl")
elo = EloRating()
if ELO_PATH.exists():
    with open(ELO_PATH, "rb") as f:
        elo.ratings = pickle.load(f)
else:
    # fallback so the API still works even if the pickle is missing
    print("⚠️  No elo_ratings.pkl found; using base ratings.")

HOME_ADVANTAGE = 65  # keep in sync with elo_model.py


# ---------- request schema ----------
class Fixture(BaseModel):
    home_team: str
    away_team: str


# ---------- prediction endpoint ----------
@app.post("/predict")
def predict(fixture: Fixture):
    """Return win/draw/away probabilities from current Elo table."""
    ra_adj = elo.get_rating(fixture.home_team) + HOME_ADVANTAGE
    rb = elo.get_rating(fixture.away_team)

    p_home = elo.expected(ra_adj, rb)
    p_away = elo.expected(rb, ra_adj)
    p_draw = max(0.0, 1 - p_home - p_away)  # quick calibration

    return {
        "home_team": fixture.home_team,
        "away_team": fixture.away_team,
        "home_win_prob": round(p_home, 3),
        "draw_prob": round(p_draw, 3),
        "away_win_prob": round(p_away, 3),
    }


# ---------- simple root route ----------
@app.get("/")
def root():
    return {"status": "ok"}
