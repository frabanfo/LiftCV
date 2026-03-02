"""
Wrapper MediaPipe Pose.

Restituisce per ogni frame un dizionario di keypoint normalizzati e in pixel.
Keypoint usati nel progetto (sottoinsieme rilevante per lo squat laterale):

  LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
  LEFT_ANKLE, RIGHT_ANKLE, LEFT_SHOULDER, RIGHT_SHOULDER,
  LEFT_WRIST, RIGHT_WRIST   ← proxy per posizione barra (fallback)
"""

import mediapipe as mp
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional

mp_pose = mp.solutions.pose

# Keypoint rilevanti — subset usato dall'analisi
LANDMARKS = {
    "left_shoulder":  mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_hip":       mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip":      mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee":      mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee":     mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle":     mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle":    mp_pose.PoseLandmark.RIGHT_ANKLE,
    "left_wrist":     mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist":    mp_pose.PoseLandmark.RIGHT_WRIST,
}


@dataclass
class PoseFrame:
    """Keypoint per un singolo frame. Coordinate in pixel (x, y) + visibilità."""
    keypoints: dict[str, tuple[float, float]]       # nome → (x_px, y_px)
    visibility: dict[str, float]                    # nome → 0.0–1.0
    raw_landmarks: object                           # oggetto mediapipe originale


class PoseEstimator:
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[PoseFrame]:
        """Ritorna PoseFrame o None se nessuna persona rilevata."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        keypoints  = {}
        visibility = {}

        for name, lm_id in LANDMARKS.items():
            lm = results.pose_landmarks.landmark[lm_id]
            keypoints[name]  = (lm.x * w, lm.y * h)
            visibility[name] = lm.visibility

        return PoseFrame(
            keypoints=keypoints,
            visibility=visibility,
            raw_landmarks=results.pose_landmarks,
        )

    def close(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
