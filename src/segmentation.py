"""
Segmentazione automatica delle fasi della ripetizione.

Le 5 fasi (§3 MVP):
  SETUP      → atleta eretto, barra sulle spalle (post START, pre discesa)
  DESCENT    → movimento verso il basso rilevato
  BOTTOM     → punto di inversione (velocità verticale barra ≈ 0)
  ASCENT     → fase concentrica (risalita)
  LOCKOUT    → ritorno alla posizione eretta finale (post RACK)

Input: serie temporale delle posizioni barra [y_px per frame].
Output: FrameRanges con indici di inizio/fine di ogni fase.

Se una fase non è rilevabile con confidenza sufficiente → flag esplicito,
nessuna validità calcolata (§3 MVP).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from src.config import CONFIDENCE_HIGH, CONFIDENCE_BORDERLINE


class Phase(Enum):
    SETUP   = auto()
    DESCENT = auto()
    BOTTOM  = auto()
    ASCENT  = auto()
    LOCKOUT = auto()


@dataclass
class PhaseSegment:
    phase: Phase
    start_frame: int
    end_frame: int
    confidence: float


@dataclass
class SegmentationResult:
    segments: list[PhaseSegment] = field(default_factory=list)
    success: bool = True
    failure_reason: Optional[str] = None

    def get(self, phase: Phase) -> Optional[PhaseSegment]:
        for s in self.segments:
            if s.phase == phase:
                return s
        return None


def segment_repetition(
    bar_y_series: np.ndarray,
    fps: float,
) -> SegmentationResult:
    """
    Segmenta la ripetizione dalla serie temporale della posizione Y della barra
    (coordinata pixel, Y cresce verso il basso → discesa = Y aumenta).

    bar_y_series: array [n_frames] con posizioni Y in pixel (None → barra occlusa).
    fps: frame rate del video.

    Algoritmo:
    1. Smoothing con Savitzky-Golay per ridurre il rumore.
    2. Derivata → velocità verticale.
    3. SETUP: frame iniziali a velocità ≈ 0.
    4. DESCENT: primo tratto a velocità > soglia (Y in aumento).
    5. BOTTOM: intorno al massimo di Y (velocità ≈ 0).
    6. ASCENT: tratto a velocità < -soglia (Y in diminuzione).
    7. LOCKOUT: frame finali a velocità ≈ 0.
    """
    if len(bar_y_series) < 30:
        return SegmentationResult(
            success=False,
            failure_reason="Ripetizione troppo breve per la segmentazione (< 30 frame).",
        )

    # Interpola eventuali None (barra occlusa brevemente)
    y = _interpolate_gaps(bar_y_series)
    if y is None:
        return SegmentationResult(
            success=False,
            failure_reason="Troppi frame senza rilevazione barra per segmentare le fasi.",
        )

    # Smoothing
    window = max(5, int(fps * 0.1) | 1)   # ~100ms, sempre dispari
    y_smooth = savgol_filter(y, window_length=window, polyorder=2)

    # Velocità (px/frame)
    velocity = np.gradient(y_smooth)

    # Soglia velocità per distinguere movimento da fermo
    vel_threshold = np.std(velocity) * 0.5

    # Indice del bottom = massimo di y_smooth (barra al punto più basso)
    bottom_idx = int(np.argmax(y_smooth))

    # Finestra di stabilità: ≥ 0.7s — abbastanza lunga da superare qualsiasi
    # velocità di discesa realistica (evita falsi positivi per decelerazione).
    min_stable = max(20, int(fps * 0.7))
    # Offset minimo dal bottom prima di iniziare la ricerca: ≥ 0.5s.
    min_phase = max(15, int(fps * 0.5))
    # Soglia di stabilità Y: 8% del range totale — abbastanza grande da
    # distinguere "fermo" (rumore ≈ 3-5 px) da "in movimento" (≥ 20 px su 0.7s).
    y_range = float(np.nanmax(y_smooth) - np.nanmin(y_smooth))
    stability_thr = max(8.0, y_range * 0.08)

    # DESCENT: l'ultimo periodo stabile di Y prima del bottom è la posizione eretta.
    # Cerchiamo a ritroso partendo da almeno min_phase frame prima del bottom.
    n_frames = len(y)
    descent_start = 0
    for i in range(bottom_idx - min_stable - min_phase, -1, -1):
        w = y_smooth[i : i + min_stable]
        if float(np.max(w) - np.min(w)) < stability_thr:
            descent_start = i + min_stable
            break

    # ASCENT: il primo periodo stabile di Y dopo il bottom è il lockout.
    # Cerchiamo in avanti partendo da almeno min_phase frame dopo il bottom.
    ascent_end = n_frames - 1
    for i in range(bottom_idx + min_phase, n_frames - min_stable + 1):
        w = y_smooth[i : i + min_stable]
        if float(np.max(w) - np.min(w)) < stability_thr:
            ascent_end = i
            break

    n = len(y)
    segments = [
        PhaseSegment(Phase.SETUP,    0,             descent_start,  _confidence_flat(velocity[:descent_start])),
        PhaseSegment(Phase.DESCENT,  descent_start, bottom_idx,     _confidence_monotone(velocity[descent_start:bottom_idx], positive=True)),
        PhaseSegment(Phase.BOTTOM,   bottom_idx,    bottom_idx,     CONFIDENCE_HIGH),
        PhaseSegment(Phase.ASCENT,   bottom_idx,    ascent_end,     _confidence_monotone(velocity[bottom_idx:ascent_end], positive=False)),
        PhaseSegment(Phase.LOCKOUT,  ascent_end,    n - 1,          _confidence_flat(velocity[ascent_end:])),
    ]

    # Controlla confidenza minima solo sulle fasi di movimento (DESCENT, BOTTOM, ASCENT).
    # SETUP e LOCKOUT possono contenere walkout/re-rack e non sono fasi analitiche.
    movement_phases = {Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT}
    min_conf = min(s.confidence for s in segments if s.phase in movement_phases)
    if min_conf < CONFIDENCE_BORDERLINE:
        return SegmentationResult(
            segments=segments,
            success=False,
            failure_reason=f"Confidenza insufficiente sulla segmentazione delle fasi ({min_conf:.0%}).",
        )

    return SegmentationResult(segments=segments, success=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def _interpolate_gaps(series) -> Optional[np.ndarray]:
    """Sostituisce None con interpolazione lineare. Ritorna None se troppi gap."""
    arr = np.array([v if v is not None else np.nan for v in series], dtype=float)
    nan_frac = np.isnan(arr).mean()
    if nan_frac > 0.2:   # >20% frame senza barra → impossibile segmentare
        return None
    # Interpolazione lineare sui NaN
    nans = np.isnan(arr)
    idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr


def _confidence_flat(velocity: np.ndarray) -> float:
    """Alta confidenza se la velocità è vicina a zero (fase statica)."""
    if len(velocity) == 0:
        return CONFIDENCE_HIGH
    std = np.std(velocity)
    return float(np.clip(1.0 - std / 5.0, 0.0, 1.0))

def _confidence_monotone(velocity: np.ndarray, positive: bool) -> float:
    """Alta confidenza se la velocità è monotona nel verso atteso."""
    if len(velocity) == 0:
        return 0.0
    sign = 1 if positive else -1
    correct = np.sum(sign * velocity > 0)
    return float(correct / len(velocity))
