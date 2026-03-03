"""
Wrapper MediaPipe Pose (Tasks API — mediapipe >= 0.10).

Restituisce per ogni frame un dizionario di keypoint normalizzati e in pixel.
Keypoint usati nel progetto (sottoinsieme rilevante per lo squat laterale):

  LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
  LEFT_ANKLE, RIGHT_ANKLE, LEFT_SHOULDER, RIGHT_SHOULDER,
  LEFT_WRIST, RIGHT_WRIST   ← proxy per posizione barra (fallback)

Il modello richiesto (pose_landmarker_lite.task) va scaricato una volta
e messo in models/ nella root del progetto.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Percorso default del modello (relativo alla root del progetto)
_DEFAULT_MODEL = Path(__file__).parent.parent.parent / "models" / "pose_landmarker_lite.task"

# Indici landmark MediaPipe Tasks API (identici all'enum PoseLandmark della vecchia API)
LANDMARK_INDICES: dict[str, int] = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_wrist":     15,
    "right_wrist":    16,
}


@dataclass
class PoseFrame:
    """Keypoint per un singolo frame. Coordinate in pixel (x, y) + visibilità."""
    keypoints:     dict[str, tuple[float, float]]   # nome → (x_px, y_px)
    visibility:    dict[str, float]                 # nome → 0.0–1.0
    raw_landmarks: object                           # lista NormalizedLandmark originale


class PoseEstimator:
    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        path = Path(model_path) if model_path else _DEFAULT_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"Modello pose non trovato: {path}\n"
                "Scaricarlo con:\n"
                "  python -c \"import urllib.request; "
                "urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_lite/float16/latest/"
                "pose_landmarker_lite.task', 'models/pose_landmarker_lite.task')\""
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_pose_presence_confidence=min_detection_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._frame_ts_ms: int = 0

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[PoseFrame]:
        """Ritorna PoseFrame o None se nessuna persona rilevata."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        self._frame_ts_ms += 33   # ~30 fps; valore esatto non critico per il tracking

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]   # persona 0
        h, w = frame_bgr.shape[:2]

        keypoints:  dict[str, tuple[float, float]] = {}
        visibility: dict[str, float] = {}

        for name, idx in LANDMARK_INDICES.items():
            lm = landmarks[idx]
            keypoints[name]  = (lm.x * w, lm.y * h)
            visibility[name] = float(lm.visibility) if lm.visibility is not None else 0.0

        return PoseFrame(
            keypoints=keypoints,
            visibility=visibility,
            raw_landmarks=landmarks,
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *_) -> None:
        self.close()
