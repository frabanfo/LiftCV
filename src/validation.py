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
    hip_y: float,
    knee_y: float,
    hip_visibility: float,
    knee_visibility: float,
) -> CriterionResult:
    """
    Verifica che il piano superiore della coscia (approssimato dall'anca)
    sia sotto il piano superiore del ginocchio nel frame di bottom.

    In coordinate pixel: Y cresce verso il basso, quindi
    'anca sotto il ginocchio' = hip_y > knee_y.

    angle_deg: angolo relativo (positivo = sotto parallela).
    """
    confidence = min(hip_visibility, knee_visibility)

    # Stima angolo (semplificata — sarà migliorata con keypoint coscia)
    delta_px = hip_y - knee_y   # positivo = anca più bassa del ginocchio

    # Convertiamo in "gradi equivalenti" rispetto alla parallela
    # Per ora usiamo delta_px normalizzato — da calibrare con dati reali
    # TODO: usare lunghezza coscia come riferimento di scala
    angle_deg = float(np.degrees(np.arctan2(delta_px, 1)))  # placeholder

    if confidence < CONFIDENCE_BORDERLINE:
        return CriterionResult(None, confidence, "Keypoint anca/ginocchio non visibili.")

    below_parallel = delta_px > 0
    in_tolerance   = abs(angle_deg - DEPTH_THRESHOLD_DEG) <= DEPTH_TOLERANCE_DEG

    if in_tolerance:
        # Zona borderline ±2°
        return CriterionResult(
            passed=None,
            confidence=confidence * 0.75,
            detail=f"Profondità al limite regolamentare ({angle_deg:+.1f}°). Non determinabile con certezza.",
        )

    passed = below_parallel
    return CriterionResult(
        passed=passed,
        confidence=confidence,
        detail=f"Angolo profondità: {angle_deg:+.1f}° rispetto parallela.",
    )


# ── Criterio 2 & 3: Lockout ──────────────────────────────────────────────────

def check_lockout(
    hip_y: float,
    knee_y: float,
    hip_visibility: float,
    knee_visibility: float,
    phase_label: str = "",
) -> CriterionResult:
    """
    Verifica estensione completa di anche e ginocchia.
    Proxy: differenza verticale anca–ginocchio minimale (entrambi allineati).
    TODO: aggiungere keypoint ginocchio angolo per misura diretta.
    """
    confidence = min(hip_visibility, knee_visibility)

    if confidence < CONFIDENCE_BORDERLINE:
        return CriterionResult(None, confidence, f"Keypoint non visibili ({phase_label}).")

    # Proxy semplice — da raffinare con angolo ginocchio esplicito
    delta = abs(hip_y - knee_y)
    extended = delta < 50   # soglia pixel — da calibrare

    return CriterionResult(
        passed=extended,
        confidence=confidence,
        detail=f"Lockout {phase_label}: {'esteso' if extended else 'non esteso'}.",
    )


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
