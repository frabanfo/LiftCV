"""
Criteri KO IPF per lo squat (§4 MVP).

Ogni funzione ritorna (passed: bool | None, confidence: float).
None = non determinabile.

Criteri KO implementati:
  1. Profondità (piano coscia < piano ginocchio al bottom)
  2. Lockout iniziale (anche e ginocchia estese nel setup)
  3. Lockout finale (anche e ginocchia estese nel lockout)
  4. Piedi (contatto continuo per tutta la ripetizione)
"""

from typing import Optional
import numpy as np

from src.config import (
    DEPTH_THRESHOLD_DEG,
    DEPTH_TOLERANCE_DEG,
    FEET_JITTER_FRAMES,
    CONFIDENCE_HIGH,
    CONFIDENCE_BORDERLINE,
    LOCKOUT_KNEE_MIN_DEG,
    LOCKOUT_HIP_MIN_DEG,
)
from src.segmentation import PhaseSegment, Phase


# ── Tipi di ritorno ──────────────────────────────────────────────────────────

class CriterionResult:
    def __init__(self, passed: Optional[bool], confidence: float, detail: str = ""):
        self.passed = passed
        self.confidence = confidence
        self.detail = detail

    @property
    def is_borderline(self) -> bool:
        return self.passed is None or (
            CONFIDENCE_BORDERLINE <= self.confidence < CONFIDENCE_HIGH
        )


# ── Criterio 1: Profondità ───────────────────────────────────────────────────

def check_depth(
    hip_x: float,
    hip_y: float,
    knee_x: float,
    knee_y: float,
    hip_visibility: float,
    knee_visibility: float,
) -> CriterionResult:
    """
    Verifica che la piega dell'anca sia sotto la sommità della rotula
    nel frame di bottom (criterio IPF parallela).

    Angolo della coscia rispetto all'orizzontale (vista laterale):
      - 0°  = anca geometricamente allo stesso livello del ginocchio
      - > 0° = sotto parallela geometrica
      - < 0° = sopra parallela geometrica

    Poiché MediaPipe restituisce il centro del giunto (non la piega dell'anca
    né la sommità della rotula), l'angolo a parallela IPF reale è circa -4° a -6°.
    DEPTH_THRESHOLD_DEG compensa questo offset: l'atleta passa il check quando
    angle_deg >= DEPTH_THRESHOLD_DEG (cioè alla parallela IPF o più in basso).
    """
    confidence = min(hip_visibility, knee_visibility)

    dx = hip_x - knee_x
    dy = hip_y - knee_y   # positivo = anca più bassa del ginocchio

    # Angolo della coscia rispetto all'orizzontale.
    # abs(dx) evita ambiguità sul verso in cui l'atleta è rivolto.
    angle_deg = float(np.degrees(np.arctan2(dy, abs(dx))))

    if confidence < CONFIDENCE_BORDERLINE:
        return CriterionResult(None, confidence, "Keypoint anca/ginocchio non visibili.")

    # L'atleta è alla/sotto parallela IPF quando l'angolo supera la soglia
    # (che include la compensazione per l'offset anatomico landmark-rotula).
    below_parallel = angle_deg >= DEPTH_THRESHOLD_DEG
    in_tolerance   = abs(angle_deg - DEPTH_THRESHOLD_DEG) <= DEPTH_TOLERANCE_DEG

    if in_tolerance:
        return CriterionResult(
            passed=None,
            confidence=confidence * 0.75,
            detail=f"Profondità al limite regolamentare ({angle_deg:+.1f}°). Non determinabile con certezza.",
        )

    return CriterionResult(
        passed=below_parallel,
        confidence=confidence,
        detail=f"Angolo profondità: {angle_deg:+.1f}° rispetto parallela.",
    )


# ── Criterio 2 & 3: Lockout ──────────────────────────────────────────────────

def check_lockout(
    shoulder: tuple[float, float],
    hip:      tuple[float, float],
    knee:     tuple[float, float],
    ankle:    tuple[float, float],
    shoulder_vis: float,
    hip_vis:      float,
    knee_vis:     float,
    ankle_vis:    float,
    phase_label: str = "",
) -> CriterionResult:
    """
    Verifica estensione completa di ginocchio e anca tramite angoli geometrici.

    Angolo al ginocchio (hip→knee→ankle): gamba tesa ≈ 180°.
    Angolo all'anca (shoulder→hip→knee): anca estesa ≈ 180°.

    Soglie da config: LOCKOUT_KNEE_MIN_DEG, LOCKOUT_HIP_MIN_DEG.
    """
    confidence = min(shoulder_vis, hip_vis, knee_vis, ankle_vis)

    if confidence < CONFIDENCE_BORDERLINE:
        return CriterionResult(None, confidence, f"Keypoint non visibili ({phase_label}).")

    knee_angle = _angle_3pt(hip,      knee, ankle)
    hip_angle  = _angle_3pt(shoulder, hip,  knee)

    knee_ok = knee_angle >= LOCKOUT_KNEE_MIN_DEG
    hip_ok  = hip_angle  >= LOCKOUT_HIP_MIN_DEG
    extended = knee_ok and hip_ok

    detail = (
        f"Lockout {phase_label}: "
        f"ginocchio {knee_angle:.1f}° ({'ok' if knee_ok else 'non esteso'}), "
        f"anca {hip_angle:.1f}° ({'ok' if hip_ok else 'non estesa'})."
    )
    return CriterionResult(passed=extended, confidence=confidence, detail=detail)


# ── helpers geometria ─────────────────────────────────────────────────────────

def _angle_3pt(
    a: tuple[float, float],
    vertex: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """Angolo in gradi al vertice formato dal segmento a–vertex–b."""
    v1 = np.array([a[0] - vertex[0], a[1] - vertex[1]], dtype=float)
    v2 = np.array([b[0] - vertex[0], b[1] - vertex[1]], dtype=float)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-9:
        return 0.0
    cos_a = np.dot(v1, v2) / norm
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


# ── Criterio 4: Piedi ────────────────────────────────────────────────────────

def check_feet(
    ankle_y_series: list[float],
    initial_ankle_y: float,
    visibility_series: list[float],
) -> CriterionResult:
    """
    Verifica contatto continuo piedi–pedana per tutta la ripetizione.
    Sollevo rilevato se ankle_y scende di più di una soglia per >= FEET_JITTER_FRAMES consecutivi.

    ankle_y_series: posizione Y della caviglia per ogni frame (Y decresce se si alza).
    initial_ankle_y: posizione di riferimento dal setup.
    """
    lift_threshold_px = 15   # pixel — da calibrare
    n = len(ankle_y_series)

    consecutive = 0
    for i, y in enumerate(ankle_y_series):
        if visibility_series[i] < CONFIDENCE_BORDERLINE:
            consecutive = 0
            continue
        if initial_ankle_y - y > lift_threshold_px:   # Y scende = piede si alza
            consecutive += 1
            if consecutive >= FEET_JITTER_FRAMES:
                return CriterionResult(
                    passed=False,
                    confidence=CONFIDENCE_HIGH,
                    detail=f"Sollevamento piedi rilevato (frame {i - FEET_JITTER_FRAMES + 1}–{i}).",
                )
        else:
            consecutive = 0

    avg_visibility = float(np.mean(visibility_series)) if visibility_series else 0.0
    return CriterionResult(passed=True, confidence=avg_visibility)
