"""
Metriche quantitative (§5 MVP).
Calcolate solo se la ripetizione è valida.

1. ROM effettivo (m) — Δ altezza barra setup → bottom
2. Velocità media concentrica (m/s) — fase ASCENT
3. Velocità di picco concentrica (m/s) — fase ASCENT
4. Deviazione orizzontale media barra (cm) — RMS rispetto asse verticale ideale
5. Angolo di profondità al bottom (°)
6. %1RM stimata — VBT curva Gonzalez-Badillo
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from src.config import VBT_A, VBT_B
from src.segmentation import SegmentationResult, Phase


@dataclass
class Metrics:
    rom_m:              Optional[float]
    avg_concentric_ms:  Optional[float]
    peak_concentric_ms: Optional[float]
    bar_deviation_cm:   Optional[float]
    depth_angle_deg:    Optional[float]
    estimated_1rm_pct:  Optional[float]


def compute_metrics(
    bar_y_px: list[Optional[float]],
    bar_x_px: list[Optional[float]],
    segmentation: SegmentationResult,
    fps: float,
    px_per_meter: Optional[float],       # calibrazione scala — None se non disponibile
    bar_weight_kg: Optional[float],
) -> Metrics:
    """
    bar_y_px / bar_x_px: posizione barra per ogni frame (None = occlusa).
    px_per_meter: fattore di scala. Se None, ROM e velocità non convertibili in unità reali.
    """

    bottom_seg  = segmentation.get(Phase.BOTTOM)
    ascent_seg  = segmentation.get(Phase.ASCENT)
    setup_seg   = segmentation.get(Phase.SETUP)

    # ── ROM ──────────────────────────────────────────────────────────────────
    # Reference: last 5 frames of SETUP = standing lockout just before descent.
    # Using the full SETUP median would include walkout (bar on J-hooks = higher
    # position), inflating ROM.
    rom_m = None
    if setup_seg and bottom_seg and px_per_meter:
        ref_start = max(setup_seg.start_frame, setup_seg.end_frame - 4)
        y_setup  = _median_valid(bar_y_px[ref_start:setup_seg.end_frame + 1])
        y_bottom = _median_valid(bar_y_px[bottom_seg.start_frame:bottom_seg.start_frame + 1])
        if y_setup is not None and y_bottom is not None:
            rom_px = abs(y_bottom - y_setup)
            rom_m  = rom_px / px_per_meter

    # ── Velocità concentrica ─────────────────────────────────────────────────
    avg_vel, peak_vel = None, None
    if ascent_seg and px_per_meter:
        y_ascent = np.array([
            v for v in bar_y_px[ascent_seg.start_frame:ascent_seg.end_frame + 1]
            if v is not None
        ], dtype=float)

        if len(y_ascent) > 5:
            n_frames = len(y_ascent)
            window = max(5, int(n_frames * 0.2) | 1)
            y_smooth = savgol_filter(y_ascent, window_length=window, polyorder=2)
            # velocità in px/frame, convertita in m/s
            vel_px_frame = np.abs(np.gradient(y_smooth))
            vel_ms = vel_px_frame * fps / px_per_meter

            avg_vel  = float(np.mean(vel_ms))
            peak_vel = float(np.max(vel_ms))

    # ── Deviazione orizzontale barra ─────────────────────────────────────────
    bar_deviation_cm = None
    if px_per_meter:
        x_valid = np.array([v for v in bar_x_px if v is not None], dtype=float)
        if len(x_valid) > 0:
            ideal_x = float(np.mean(x_valid))
            rms_px  = float(np.sqrt(np.mean((x_valid - ideal_x) ** 2)))
            bar_deviation_cm = rms_px / px_per_meter * 100

    # ── %1RM (VBT — Gonzalez-Badillo) ────────────────────────────────────────
    estimated_1rm_pct = None
    if bar_weight_kg is not None and avg_vel is not None:
        # Formula: %1RM = 100 * (1 - A * v^B)
        estimated_1rm_pct = 100.0 * (1.0 - VBT_A * (avg_vel ** VBT_B))
        estimated_1rm_pct = float(np.clip(estimated_1rm_pct, 0, 100))

    # ── Angolo profondità (placeholder) ─────────────────────────────────────
    # Calcolato in validation.check_depth — qui lo riceviamo dall'esterno
    depth_angle_deg = None  # impostato dal pipeline dopo validation

    return Metrics(
        rom_m=rom_m,
        avg_concentric_ms=avg_vel,
        peak_concentric_ms=peak_vel,
        bar_deviation_cm=bar_deviation_cm,
        depth_angle_deg=depth_angle_deg,
        estimated_1rm_pct=estimated_1rm_pct,
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def _median_valid(series) -> Optional[float]:
    vals = [v for v in series if v is not None]
    return float(np.median(vals)) if vals else None
