"""
Bar tracking: ROI custom su colore/forma + Kalman filter per occlusioni brevi.

Strategia MVP:
1. L'utente definisce (o il sistema propone) una ROI iniziale sul piatto/collare della barra.
2. Il tracking segue la ROI frame per frame.
3. Se il centro non viene trovato per <= BAR_OCCLUSION_MAX_FRAMES → Kalman predice.
4. Se supera la soglia → le metriche barra vengono marcate N/D (nessuna interpolazione silenziosa).

Output per frame: posizione barra (x, y) in pixel, o None se non disponibile.
"""

import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from typing import Optional

from src.config import BAR_OCCLUSION_MAX_FRAMES


class BarTracker:
    def __init__(self):
        self._kf = self._build_kalman()
        self._occlusion_count = 0
        self._initialized = False
        self.exceeded_occlusion_limit = False  # flag per metriche N/D

    # ── public ───────────────────────────────────────────────────────────────

    def initialize(self, x: float, y: float) -> None:
        """Prima rilevazione confermata dall'utente o dal sistema."""
        self._kf.x = np.array([[x], [y], [0.], [0.]])
        self._initialized = True
        self._occlusion_count = 0

    def update(self, detection: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
        """
        Aggiorna con la rilevazione del frame corrente.
        detection: (x, y) in pixel, o None se non trovata.
        Ritorna posizione stimata (x, y) o None se superata la soglia occlusione.
        """
        if not self._initialized:
            return None

        if detection is not None:
            self._kf.predict()
            self._kf.update(np.array([[detection[0]], [detection[1]]]))
            self._occlusion_count = 0
        else:
            self._occlusion_count += 1
            if self._occlusion_count > BAR_OCCLUSION_MAX_FRAMES:
                self.exceeded_occlusion_limit = True
                return None
            self._kf.predict()  # predizione senza misurazione

        x, y = float(self._kf.x[0]), float(self._kf.x[1])
        return (x, y)

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_kalman() -> KalmanFilter:
        """
        Stato: [x, y, vx, vy] — moto a velocità costante.
        Misura: [x, y]
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)

        dt = 1.0  # 1 frame
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1,  0],
                         [0, 0, 0,  1]], dtype=float)

        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]], dtype=float)

        kf.R  *= 10     # rumore di misura
        kf.P  *= 100    # incertezza iniziale
        kf.Q  *= 0.1    # rumore di processo (barra si muove lentamente)

        return kf


def detect_bar_in_roi(frame_bgr: np.ndarray, roi: tuple[int, int, int, int]) -> Optional[tuple[float, float]]:
    """
    Rilevazione del centro della barra nella ROI tramite colore (acciaio/cromato)
    e forma circolare (piatto/collare).

    roi: (x, y, w, h) in pixel

    TODO: calibrare soglie HSV su dati reali.
    """
    x, y, w, h = roi
    crop = frame_bgr[y:y+h, x:x+w]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Range colore acciaio cromato — da calibrare
    lower = np.array([0, 0, 150])
    upper = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Centroide della maschera
    M = cv2.moments(mask)
    if M["m00"] < 100:   # area troppo piccola → non rilevato
        return None

    cx = M["m10"] / M["m00"] + x
    cy = M["m01"] / M["m00"] + y
    return (cx, cy)
