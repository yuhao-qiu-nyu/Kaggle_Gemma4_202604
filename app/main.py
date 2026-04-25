"""
ASL Sign Language Recognition — FastAPI service

Endpoints
---------
GET  /health              → liveness check
POST /predict             → accept landmark JSON, return top-k
POST /predict_from_file   → accept a parquet path on disk (dev/test convenience)
POST /coach               → predict + Gemma 4 coaching feedback in one call
POST /coach_from_file     → same, but from a parquet file on disk
"""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .llm import CoachLLM
from .mediapipe_extractor import extract_landmarks_from_video
from .model import ASLEnsemble
from .preprocess import ROWS_PER_FRAME
from .schemas import (
    CoachFromFileRequest,
    CoachRequest,
    CoachResponse,
    PredictFromFileRequest,
    PredictRequest,
    PredictResponse,
    VideoCoachResponse,
)

# ---------------------------------------------------------------------------
# Config — override via environment variables if needed
# ---------------------------------------------------------------------------
BASE_DIR = Path(os.getenv(
    "ASL_BASE_DIR",
    "/Volumes/senzu/LLMPROJECTs/ASL",
))

WEIGHT_DIR = BASE_DIR / "tensorflow" / "islr-models"
SIGN_MAP = (
    BASE_DIR / "datasets" / "sohier" / "461054610546105"
    / "versions" / "5" / "sign_to_prediction_index_map.json"
)

WEIGHT_FILES = sorted(WEIGHT_DIR.glob("islr-fp16-192-8-seed*-foldall-last.h5"))

ensemble: ASLEnsemble | None = None
coach_llm: CoachLLM | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ensemble, coach_llm
    if not WEIGHT_FILES:
        raise RuntimeError(f"No .h5 weight files found in {WEIGHT_DIR}")
    if not SIGN_MAP.exists():
        raise RuntimeError(f"sign_to_prediction_index_map.json not found at {SIGN_MAP}")

    print(f"[startup] Loading {len(WEIGHT_FILES)} model weight(s) …")
    ensemble = ASLEnsemble(
        weight_paths=[str(p) for p in WEIGHT_FILES],
        sign_map_path=str(SIGN_MAP),
    )
    print("[startup] Models loaded ✓")

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        coach_llm = CoachLLM(api_key=gemini_key)
        print("[startup] Gemma 4 coach initialized ✓")
    else:
        print("[startup] GEMINI_API_KEY not set — /coach endpoints disabled")

    yield
    ensemble = None
    coach_llm = None


app = FastAPI(
    title="ASL Sign Recognition API",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _landmarks_from_frames(frames) -> np.ndarray:
    raw = np.array([frame.landmarks for frame in frames], dtype=np.float32)
    if raw.shape[1:] != (ROWS_PER_FRAME, 3):
        raise ValueError(f"Expected each frame to be ({ROWS_PER_FRAME}, 3), got {raw.shape[1:]}")
    return raw


def _landmarks_from_parquet(parquet_path: str) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=["x", "y", "z"])
    n_frames = len(df) // ROWS_PER_FRAME
    return df.values.reshape(n_frames, ROWS_PER_FRAME, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Endpoints — prediction only
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": ensemble is not None,
        "num_models": len(ensemble.models) if ensemble else 0,
        "coach_available": coach_llm is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Accept landmark frames as JSON and return sign prediction."""
    if ensemble is None:
        raise HTTPException(503, "Models not loaded yet")
    try:
        raw = _landmarks_from_frames(req.frames)
    except Exception as e:
        raise HTTPException(422, f"Failed to parse landmarks: {e}")
    return PredictResponse(**ensemble.predict(raw, topk=req.topk))


@app.post("/predict_from_file", response_model=PredictResponse)
def predict_from_file(req: PredictFromFileRequest):
    """Load a .parquet landmark file from disk and return sign prediction."""
    if ensemble is None:
        raise HTTPException(503, "Models not loaded yet")
    if not os.path.isfile(req.parquet_path):
        raise HTTPException(404, f"File not found: {req.parquet_path}")
    try:
        raw = _landmarks_from_parquet(req.parquet_path)
    except Exception as e:
        raise HTTPException(422, f"Failed to read parquet: {e}")
    return PredictResponse(**ensemble.predict(raw, topk=req.topk))


# ---------------------------------------------------------------------------
# Endpoints — coach (predict + Gemma 4 feedback)
# ---------------------------------------------------------------------------

@app.post("/coach", response_model=CoachResponse)
def coach(req: CoachRequest):
    """Predict sign + get Gemma 4 coaching feedback in one call."""
    if ensemble is None:
        raise HTTPException(503, "Models not loaded yet")
    if coach_llm is None:
        raise HTTPException(503, "GEMINI_API_KEY not set — coach unavailable")

    try:
        raw = _landmarks_from_frames(req.frames)
    except Exception as e:
        raise HTTPException(422, f"Failed to parse landmarks: {e}")

    pred = ensemble.predict(raw, topk=req.topk)

    try:
        llm_result = coach_llm.get_coach_feedback(
            pred_result=pred,
            user_goal=req.user_goal,
            history_errors=req.history_errors,
        )
    except Exception as e:
        raise HTTPException(502, f"Gemma 4 API error: {e}")

    return CoachResponse(**pred, **llm_result)


@app.post("/coach_from_file", response_model=CoachResponse)
def coach_from_file(req: CoachFromFileRequest):
    """Parquet file version of /coach — handy for testing."""
    if ensemble is None:
        raise HTTPException(503, "Models not loaded yet")
    if coach_llm is None:
        raise HTTPException(503, "GEMINI_API_KEY not set — coach unavailable")
    if not os.path.isfile(req.parquet_path):
        raise HTTPException(404, f"File not found: {req.parquet_path}")

    try:
        raw = _landmarks_from_parquet(req.parquet_path)
    except Exception as e:
        raise HTTPException(422, f"Failed to read parquet: {e}")

    pred = ensemble.predict(raw, topk=req.topk)

    try:
        llm_result = coach_llm.get_coach_feedback(
            pred_result=pred,
            user_goal=req.user_goal,
            history_errors=req.history_errors,
        )
    except Exception as e:
        raise HTTPException(502, f"Gemma 4 API error: {e}")

    return CoachResponse(**pred, **llm_result)


# ---------------------------------------------------------------------------
# Endpoints — video coach (upload .mp4 → MediaPipe → predict → Gemma 4)
# ---------------------------------------------------------------------------

@app.post("/coach_video", response_model=VideoCoachResponse)
def coach_video(
    video: UploadFile = File(..., description="An .mp4 video of a sign attempt"),
    topk: int = Form(default=5, ge=1, le=250),
    user_goal: Optional[str] = Form(default=None),
    history_errors: Optional[str] = Form(
        default=None,
        description="Comma-separated list of previously confused signs, e.g. 'cloud,grandma'",
    ),
):
    """
    Full pipeline in one call:
    upload video -> MediaPipe landmarks -> ASL model -> Gemma 4 coach feedback.
    """
    if ensemble is None:
        raise HTTPException(503, "Models not loaded yet")
    if coach_llm is None:
        raise HTTPException(503, "GEMINI_API_KEY not set — coach unavailable")

    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video.file.read())
        tmp_path = tmp.name

    try:
        raw = extract_landmarks_from_video(tmp_path)
    except FileNotFoundError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(422, f"MediaPipe extraction failed: {e}")
    finally:
        os.unlink(tmp_path)

    if raw.shape[0] == 0:
        raise HTTPException(422, "No frames could be extracted from the video")

    num_frames = raw.shape[0]
    pred = ensemble.predict(raw, topk=topk)

    errors_list = [s.strip() for s in history_errors.split(",")] if history_errors else None
    try:
        llm_result = coach_llm.get_coach_feedback(
            pred_result=pred,
            user_goal=user_goal,
            history_errors=errors_list,
        )
    except Exception as e:
        raise HTTPException(502, f"Gemma 4 API error: {e}")

    return VideoCoachResponse(**pred, **llm_result, num_frames=num_frames)
