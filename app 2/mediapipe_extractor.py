"""
MediaPipe Holistic Landmarker — extract 543 landmarks per frame from video.

Landmark order (must match the ASL model's training data):
  face:       468 points  (indices 0–467)
  left_hand:   21 points  (indices 468–488)
  pose:        33 points  (indices 489–521)
  right_hand:  21 points  (indices 522–542)
  Total:      543 points
"""

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROWS_PER_FRAME = 543

_LANDMARK_GROUPS = [
    ("face_landmarks", 468),
    ("left_hand_landmarks", 21),
    ("pose_landmarks", 33),
    ("right_hand_landmarks", 21),
]

_DEFAULT_MODEL_PATH = os.getenv(
    "HOLISTIC_MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "holistic_landmarker.task"),
)


def _flatten_group(landmarks, expected_count: int) -> np.ndarray:
    """Convert one landmark group into (expected_count, 3) array, NaN-fill if missing."""
    if landmarks:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        if len(pts) == expected_count:
            return pts
    return np.full((expected_count, 3), np.nan, dtype=np.float32)


def extract_landmarks_from_video(
    video_path: str,
    model_path: Optional[str] = None,
) -> np.ndarray:
    """
    Process a video file and return landmarks as (T, 543, 3) float32 array.

    Raises FileNotFoundError if the video or model file doesn't exist.
    Returns an empty (0, 543, 3) array if no frames could be read.
    """
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = model_path or _DEFAULT_MODEL_PATH
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Holistic model not found at {model_path}. "
            "Download it with: wget -O holistic_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/"
            "holistic_landmarker/holistic_landmarker/float16/latest/"
            "holistic_landmarker.task"
        )

    options = vision.HolisticLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        output_face_blendshapes=False,
        min_face_detection_confidence=0.3,
        min_hand_landmarks_confidence=0.3,
        min_pose_landmarks_confidence=0.3,
    )
    detector = vision.HolisticLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            parts = []
            for attr, count in _LANDMARK_GROUPS:
                lm = getattr(result, attr, None)
                parts.append(_flatten_group(lm, count))
            frames.append(np.concatenate(parts, axis=0))  # (543, 3)
    finally:
        cap.release()
        detector.close()

    if not frames:
        return np.empty((0, ROWS_PER_FRAME, 3), dtype=np.float32)

    arr = np.stack(frames, axis=0)  # (T, 543, 3)

    # Linear interpolation across frames to fill NaN gaps (same as kaggle script)
    for col in range(ROWS_PER_FRAME):
        for dim in range(3):
            series = arr[:, col, dim]
            nans = np.isnan(series)
            if nans.any() and not nans.all():
                valid = np.where(~nans)[0]
                arr[:, col, dim] = np.interp(
                    np.arange(len(series)), valid, series[valid]
                )

    return arr.astype(np.float32)
