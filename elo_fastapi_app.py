from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Football Elo Predictor", version="0.1.0")


# ---------- health-check ----------
@app.get("/")
def root():
    return {"status": "ok"}


# ---------- request body schema ----------
class Fixture(BaseModel):
    home_team: str
    away_team: str


# ---------- prediction endpoint ----------
@app.post("/predict")
def predict(fixture: Fixture):
    """Return placeholder probabilities (we'll wire real Elo later)."""
    return {
        "home_team": fixture.home_team,
        "away_team": fixture.away_team,
        "home_win_prob": 0.5,
        "draw_prob": 0.0,
        "away_win_prob": 0.5,
    }


#from fastapi import FastAPI
#app = FastAPI(title="Football Elo Predictor", version="0.1.0")

#@app.get("/")
#def root():
    #return {"status": "ok"}