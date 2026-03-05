"""
Segmentazione automatica delle fasi della ripetizione.

Le 5 fasi (S3 MVP):
  SETUP      -> atleta eretto, barra sulle spalle (post START, pre discesa)
  DESCENT    -> movimento verso il basso rilevato
  BOTTOM     -> punto di inversione (velocita verticale barra ~ 0)
  ASCENT     -> fase concentrica (risalita)
  LOCKOUT    -> ritorno alla posizione eretta finale (post RACK)

Input: serie temporale delle posizioni barra [y_px per frame].
Output: FrameRanges con indici di inizio/fine di ogni fase.

Se una fase non e rilevabile con confidenza sufficiente -> flag esplicito,
nessuna validita calcolata (S3 MVP).
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
    px_per_meter: Optional[float] = None,
) -> SegmentationResult:
    """
    Segmenta la ripetizione dalla serie temporale della posizione Y della barra
    (coordinata pixel, Y cresce verso il basso -> discesa = Y aumenta).

    bar_y_series: array [n_frames] con posizioni Y in pixel (None -> barra occlusa).
    fps: frame rate del video.
    px_per_meter: fattore di conversione pixel/metro (opzionale).

    Algoritmo "from-the-hole":
    1. Smoothing Savitzky-Golay (finestra ~0.5s) per ridurre il rumore dei keypoint.
    2. Derivata prima -> velocita verticale sulla curva liscia.
    3. BOTTOM: massimo globale di y_smooth (punto piu basso della traiettoria).
    4. SETUP end: primo frame da 0 in cui la velocita e positiva sostenuta >= min_sustained frame.
    5. LOCKOUT start: primo frame dopo BOTTOM in cui |velocita| < v_thresh sostenuto >= min_sustained frame.
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

    # ── Stage 1: Signal cleaning ─────────────────────────────────────────────
    window = max(5, int(fps * 0.5) | 1)  # ~0.5s, always odd
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    y_smooth = savgol_filter(y, window_length=window, polyorder=2)

    velocity = np.gradient(y_smooth)

    # ── Stage 2: Find bottom (global minimum, i.e. max Y) ────────────────────
    bottom_idx = int(np.argmax(y_smooth))

    # ── Stage 3: Phase segmentation via velocity threshold ────────────────────
    # Adaptive velocity threshold
    if px_per_meter is not None:
        v_thresh = 0.02 * px_per_meter / fps  # 0.02 m/s -> px/frame
    else:
        v_thresh = max(0.5, float(np.percentile(np.abs(velocity), 25)))

    min_sustained = max(5, int(fps * 0.2))  # ~200ms

    # descent_start: first sustained positive velocity before bottom
    descent_start = _find_sustained_crossing(
        velocity, v_thresh,
        search_start=0,
        search_end=bottom_idx,
        min_sustained=min_sustained,
    )

    # ascent_end: end of concentric phase, found via peak velocity.
    # Strategy: find peak upward velocity (argmin in [bottom, bottom+5s]),
    # then find the first frame after that peak where the bar is no longer
    # moving significantly upward (velocity >= -v_thresh).
    # Using peak-based detection instead of sustained-stability avoids the
    # re-rack problem: sustained stability never triggers if re-racking starts
    # immediately after lockout, pushing the fallback to n-1.
    n = len(y)

    vel_limit = min(bottom_idx + int(fps * 5), n)
    peak_conc_idx = bottom_idx + int(np.argmin(velocity[bottom_idx:vel_limit]))

    ascent_end = n - 1
    for i in range(peak_conc_idx, vel_limit):
        if velocity[i] >= -v_thresh:
            ascent_end = i
            break
    segments = [
        PhaseSegment(Phase.SETUP,    0,             descent_start, _confidence_flat(velocity[:descent_start], v_thresh)),
        PhaseSegment(Phase.DESCENT,  descent_start, bottom_idx,    _confidence_monotone(velocity[descent_start:bottom_idx], positive=True)),
        PhaseSegment(Phase.BOTTOM,   bottom_idx,    bottom_idx,    CONFIDENCE_HIGH),
        PhaseSegment(Phase.ASCENT,   bottom_idx,    ascent_end,    _confidence_monotone(velocity[bottom_idx:ascent_end], positive=False)),
        PhaseSegment(Phase.LOCKOUT,  ascent_end,    n - 1,         _confidence_flat(velocity[ascent_end:], v_thresh)),
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


# -- helpers ------------------------------------------------------------------

def _interpolate_gaps(series) -> Optional[np.ndarray]:
    """Sostituisce None con interpolazione lineare. Ritorna None se troppi gap."""
    arr = np.array([v if v is not None else np.nan for v in series], dtype=float)
    nan_frac = np.isnan(arr).mean()
    if nan_frac > 0.2:   # >20% frame senza barra -> impossibile segmentare
        return None
    # Interpolazione lineare sui NaN
    nans = np.isnan(arr)
    idx = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr


def _find_sustained_crossing(
    velocity: np.ndarray,
    v_thresh: float,
    search_start: int,
    search_end: int,
    min_sustained: int,
) -> int:
    """First index from which velocity > v_thresh for min_sustained consecutive frames."""
    count = 0
    start_candidate = search_start
    for i in range(search_start, min(search_end, len(velocity))):
        if velocity[i] > v_thresh:
            if count == 0:
                start_candidate = i
            count += 1
            if count >= min_sustained:
                return start_candidate
        else:
            count = 0
    return search_start  # fallback: no pre-motion static phase found


def _find_sustained_stable(
    velocity: np.ndarray,
    v_thresh: float,
    search_start: int,
    search_end: int,
    min_sustained: int,
) -> int:
    """First index from which |velocity| < v_thresh for min_sustained consecutive frames."""
    count = 0
    start_candidate = search_start
    for i in range(search_start, min(search_end, len(velocity))):
        if abs(velocity[i]) < v_thresh:
            if count == 0:
                start_candidate = i
            count += 1
            if count >= min_sustained:
                return start_candidate
        else:
            count = 0
    return min(search_end - 1, len(velocity) - 1)  # fallback: no lockout found


def _confidence_flat(velocity: np.ndarray, v_thresh: float = 5.0) -> float:
    """Alta confidenza se la velocita e vicina a zero (fase statica)."""
    if len(velocity) == 0:
        return CONFIDENCE_HIGH
    std = float(np.std(velocity))
    return float(np.clip(1.0 - std / (v_thresh * 3.0), 0.0, 1.0))


def _confidence_monotone(velocity: np.ndarray, positive: bool) -> float:
    """Alta confidenza se la velocita e monotona nel verso atteso."""
    if len(velocity) == 0:
        return 0.0
    sign = 1 if positive else -1
    correct = np.sum(sign * velocity > 0)
    return float(correct / len(velocity))
