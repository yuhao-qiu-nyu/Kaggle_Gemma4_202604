"""Pydantic models for request / response validation."""

from typing import List, Optional

from pydantic import BaseModel, Field


class LandmarkFrame(BaseModel):
    """One frame of 543 landmarks, each with x/y/z."""
    landmarks: List[List[float]] = Field(
        ...,
        description="List of 543 [x, y, z] points for this frame",
    )


class PredictRequest(BaseModel):
    """
    Accept landmark data as a list of frames.
    Each frame has 543 landmarks × 3 coordinates.
    """
    frames: List[LandmarkFrame] = Field(
        ...,
        description="Sequence of frames, each containing 543 landmarks",
    )
    topk: int = Field(default=5, ge=1, le=250, description="Number of top predictions to return")


class PredictFromFileRequest(BaseModel):
    """Accept a path to a parquet file on disk (for testing convenience)."""
    parquet_path: str = Field(..., description="Absolute path to a .parquet landmark file")
    topk: int = Field(default=5, ge=1, le=250)


class TopKItem(BaseModel):
    rank: int
    label: str
    prob: float


class PredictResponse(BaseModel):
    predicted_label: str
    predicted_prob: float
    confidence: str
    topk_predictions: List[TopKItem]


# ---------------------------------------------------------------------------
# Coach (ASL predict + Gemma 4 feedback in one call)
# ---------------------------------------------------------------------------

class CoachRequest(BaseModel):
    """Landmarks + optional coaching context."""
    frames: List[LandmarkFrame] = Field(
        ...,
        description="Sequence of frames, each containing 543 landmarks",
    )
    topk: int = Field(default=5, ge=1, le=250)
    user_goal: Optional[str] = Field(default=None, description="e.g. 'learn family-related ASL vocabulary'")
    history_errors: Optional[List[str]] = Field(default=None, description="Previously confused signs")


class CoachFromFileRequest(BaseModel):
    """Parquet path version for testing."""
    parquet_path: str = Field(..., description="Absolute path to a .parquet landmark file")
    topk: int = Field(default=5, ge=1, le=250)
    user_goal: Optional[str] = Field(default=None)
    history_errors: Optional[List[str]] = Field(default=None)


class CoachResponse(BaseModel):
    predicted_label: str
    predicted_prob: float
    confidence: str
    topk_predictions: List[TopKItem]
    coach_feedback: str
    gemma_payload: dict


# ---------------------------------------------------------------------------
# Video coach (upload .mp4 → MediaPipe → predict → Gemma 4)
# ---------------------------------------------------------------------------

class VideoCoachResponse(CoachResponse):
    """Same as CoachResponse + metadata about the extracted video."""
    num_frames: int
