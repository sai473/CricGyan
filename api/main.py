"""
FastAPI app: serves the cricgnaan HTML UI and /api/predict (real ensemble models).
Run: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""
import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

STATIC = os.path.join(ROOT, "static")


class PredictIn(BaseModel):
    team_a: str
    team_b: str
    venue: str
    toss_winner: str
    toss_decision: str = "bat"
    playoff: bool = False
    over: int = 0
    score_1st: int = 0
    score_2nd: int = 0
    wkts_2nd: int = 0
    batter_sr: float = 120.0
    partnership: float = 6.0
    bowler_eco: float = 8.0

    @field_validator("toss_decision")
    @classmethod
    def _dec(cls, v: str) -> str:
        if v not in ("bat", "field"):
            raise ValueError("toss_decision must be bat or field")
        return v


app = FastAPI(title="CricGyaan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_models():
    from api.inference import load_models_cached

    load_models_cached()


@app.get("/health")
def health():
    from api.inference import load_models_cached

    m = load_models_cached()
    return {"ok": True, "models_loaded": m is not None and m[0] is not None}


@app.post("/api/predict")
def api_predict(body: PredictIn):
    from api.inference import run_predict

    try:
        return run_predict(body.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/")
def index():
    path = os.path.join(STATIC, "index.html")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="static/index.html missing")
    return FileResponse(path, media_type="text/html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
