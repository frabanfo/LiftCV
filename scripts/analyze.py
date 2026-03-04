"""
Entry point MVP — LiftCV.

Uso:
    python scripts/analyze.py <video_path>

Il programma chiede interattivamente i dati utente (peso bilanciere obbligatorio,
peso corporeo e 1RM storico opzionali) e stampa il report a console.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Aggiunge la root del progetto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io.video import load_video, iter_frames
from src.io.output import AnalysisResult, print_report
from src.tracking.pose import PoseEstimator, PoseFrame
from src.tracking.bar_tracker import BarTracker, detect_bar_in_roi
from src.segmentation import segment_repetition, Phase
from src.validation import check_depth, check_lockout, check_feet, CriterionResult
from src.metrics import compute_metrics
from src import config
from src.config import TIBIA_HEIGHT_RATIO, CONFIDENCE_BORDERLINE


def main():
    parser = argparse.ArgumentParser(description="LiftCV — Analisi squat da video laterale")
    parser.add_argument("video", help="Percorso del video da analizzare")
    parser.add_argument("--debug", action="store_true", help="Stampa info diagnostiche sul segnale barra e segmentazione")
    args = parser.parse_args()

    # ── Caricamento video ────────────────────────────────────────────────────
    try:
        cap, meta = load_video(args.video)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERRORE INPUT: {e}")
        sys.exit(1)

    print(f"\nVideo caricato: {meta.path.name}")
    print(f"  {meta.frame_count} frame  |  {meta.fps:.1f} fps  |  {meta.width}×{meta.height}px")

    # ── Dati utente ──────────────────────────────────────────────────────────
    bar_weight_kg     = _ask_float("Peso sul bilanciere (kg) [obbligatorio]: ", required=True)
    height_m          = _ask_height("Altezza atleta (m, es. 1.75) [obbligatorio]: ")
    body_weight_kg    = _ask_float("Peso corporeo (kg) [invio per saltare]: ", required=False)    # noqa: F841  (reserved for future use)
    historical_1rm_kg = _ask_float("1RM storico (kg) [invio per saltare]: ",   required=False)   # noqa: F841

    # ── Scan frame: pose + barra ──────────────────────────────────────────────
    print("\nAnalisi in corso...")

    pose_frames: list[Optional[PoseFrame]] = []
    bar_x_px:    list[Optional[float]]     = []
    bar_y_px:    list[Optional[float]]     = []

    bar_tracker    = BarTracker()
    bar_initialized = False

    detection_count = 0
    total_frames    = 0

    with PoseEstimator() as pose_est:
        for frame_bgr in iter_frames(cap):
            pf = pose_est.process_frame(frame_bgr)
            pose_frames.append(pf)

            detection = _detect_bar(pf, frame_bgr, meta.width, meta.height)
            total_frames += 1
            if detection is not None:
                detection_count += 1

            if not bar_initialized and detection is not None:
                bar_tracker.initialize(detection[0], detection[1])
                bar_initialized = True

            pos = bar_tracker.update(detection) if bar_initialized else None
            bar_x_px.append(pos[0] if pos else None)
            bar_y_px.append(pos[1] if pos else None)

    cap.release()

    if args.debug:
        pose_detected = sum(1 for pf in pose_frames if pf is not None)
        print(f"\n[DEBUG] Pose rilevata:   {pose_detected}/{total_frames} frame ({pose_detected/total_frames:.0%})")
        print(f"[DEBUG] Barra rilevata:  {detection_count}/{total_frames} frame ({detection_count/total_frames:.0%})")

    if not bar_initialized:
        _print_rejection(
            "Barra non rilevata nel video. "
            "Assicurarsi che la barra sia visibile e illuminata correttamente.",
            bar_weight_kg,
        )
        return

    # ── Segmentazione ─────────────────────────────────────────────────────────
    bar_y_arr = np.array([y if y is not None else np.nan for y in bar_y_px])

    if args.debug:
        valid_y = bar_y_arr[~np.isnan(bar_y_arr)]
        bottom_idx_dbg = int(np.nanargmax(bar_y_arr))
        print(f"\n[DEBUG] Segnale barra Y:")
        print(f"  Range:      {valid_y.min():.0f}–{valid_y.max():.0f} px  (Δ={valid_y.max()-valid_y.min():.0f} px)")
        print(f"  NaN:        {np.isnan(bar_y_arr).sum()}/{len(bar_y_arr)} frame")
        print(f"  Bottom idx: frame {bottom_idx_dbg}  (Y={bar_y_arr[bottom_idx_dbg]:.0f} px)")
        print(f"  Inizio:     Y={valid_y[0]:.0f} px  |  Fine: Y={valid_y[-1]:.0f} px")

    seg = segment_repetition(bar_y_arr, meta.fps)

    if args.debug:
        status = "OK" if seg.success else "FALLITA"
        print(f"\n[DEBUG] Segmentazione — {status}")
        for s in seg.segments:
            flag = "  ← PROBLEMA" if s.confidence < CONFIDENCE_BORDERLINE else ""
            print(f"  {s.phase.name:<10} frame {s.start_frame:>4}–{s.end_frame:<4}  conf={s.confidence:.0%}{flag}")
        if not seg.success:
            print(f"  Motivo: {seg.failure_reason}")

    if not seg.success:
        _print_rejection(seg.failure_reason, bar_weight_kg)
        return

    bottom_seg  = seg.get(Phase.BOTTOM)
    setup_seg   = seg.get(Phase.SETUP)
    lockout_seg = seg.get(Phase.LOCKOUT)

    # ── Calibrazione scala ────────────────────────────────────────────────────
    px_per_meter = _compute_px_per_meter(pose_frames, setup_seg, height_m)
    if px_per_meter is None:
        print("  Avviso: scala non calcolabile (tibia non visibile nel setup) — metriche in m/s = N/D.")

    # ── Validazione criteri KO ────────────────────────────────────────────────
    depth_result, depth_angle = _check_depth_at(pose_frames, bottom_seg.start_frame)
    init_lockout  = _check_lockout_at(pose_frames, setup_seg.end_frame,    "iniziale")
    final_lockout = _check_lockout_at(pose_frames, lockout_seg.start_frame, "finale")
    feet_result   = _check_feet_series(pose_frames, setup_seg)

    # ── Metriche ─────────────────────────────────────────────────────────────
    metrics = compute_metrics(
        bar_y_px=bar_y_px,
        bar_x_px=bar_x_px,
        segmentation=seg,
        fps=meta.fps,
        px_per_meter=px_per_meter,
        bar_weight_kg=bar_weight_kg,
    )

    # ── Validità e confidenza complessiva ─────────────────────────────────────
    ko_criteria = [depth_result, init_lockout, final_lockout, feet_result]
    valid, confidence = _aggregate_validity(ko_criteria)

    bar_stability_ok: Optional[bool] = (
        metrics.bar_deviation_cm < 5.0
        if metrics.bar_deviation_cm is not None else None
    )

    result = AnalysisResult(
        valid=valid,
        confidence=confidence,
        depth_ok=depth_result.passed,
        depth_angle_deg=depth_angle,
        initial_lockout_ok=init_lockout.passed,
        final_lockout_ok=final_lockout.passed,
        feet_ok=feet_result.passed,
        rom_m=metrics.rom_m,
        avg_concentric_ms=metrics.avg_concentric_ms,
        peak_concentric_ms=metrics.peak_concentric_ms,
        bar_deviation_cm=metrics.bar_deviation_cm,
        bar_weight_kg=bar_weight_kg,
        estimated_1rm_pct=metrics.estimated_1rm_pct,
        bar_stability_ok=bar_stability_ok,
        symmetry_pct=None,
        symmetry_confidence=None,
    )
    print_report(result)


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _detect_bar(
    pf: Optional[PoseFrame],
    frame_bgr,
    width: int,
    height: int,
) -> Optional[tuple[float, float]]:
    """
    Stima la posizione della barra nel frame corrente.

    Strategia:
    1. Calcola il punto medio dei polsi (proxy barra in back-squat).
    2. Prova detect_bar_in_roi (colore cromato) in una ROI intorno a quel punto.
    3. Se fallisce ma i polsi sono visibili, usa il loro punto medio come fallback.
    4. Se i polsi non sono visibili, usa il punto medio delle spalle.
    """
    if pf is None:
        return None

    lw = pf.keypoints.get("left_wrist")
    rw = pf.keypoints.get("right_wrist")
    vis_lw = pf.visibility.get("left_wrist", 0.0)
    vis_rw = pf.visibility.get("right_wrist", 0.0)

    if lw is not None and rw is not None:
        mx, my = (lw[0] + rw[0]) / 2, (lw[1] + rw[1]) / 2
        wrist_visible = (vis_lw + vis_rw) / 2 > 0.5
    else:
        # Fallback: punto medio spalle
        ls = pf.keypoints.get("left_shoulder")
        rs = pf.keypoints.get("right_shoulder")
        if ls is None or rs is None:
            return None
        mx, my = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
        wrist_visible = False

    r = 60
    roi = (
        max(0, int(mx - r)),
        max(0, int(my - r)),
        min(width  - max(0, int(mx - r)), 2 * r),
        min(height - max(0, int(my - r)), 2 * r),
    )
    detected = detect_bar_in_roi(frame_bgr, roi)
    if detected is not None:
        return detected

    return (mx, my) if wrist_visible else None


def _dominant_side(pf: PoseFrame, landmarks: list[str]) -> str:
    """Ritorna 'left' o 'right': il lato con la visibilità maggiore."""
    left_vis  = sum(pf.visibility.get(f"left_{lm}",  0.0) for lm in landmarks)
    right_vis = sum(pf.visibility.get(f"right_{lm}", 0.0) for lm in landmarks)
    return "left" if left_vis >= right_vis else "right"


def _compute_px_per_meter(
    pose_frames: list[Optional[PoseFrame]],
    setup_seg,
    height_m: float,
) -> Optional[float]:
    """
    Stima px/metro usando la tibia come riferimento antropometrico.
    Lunghezza tibia stimata = altezza × TIBIA_HEIGHT_RATIO (Drillis & Contini).
    Misura la tibia in pixel sui frame stabili del setup e ne prende la mediana.
    """
    tibia_m = height_m * TIBIA_HEIGHT_RATIO

    side = "left"
    for pf in pose_frames[setup_seg.start_frame:setup_seg.end_frame + 1]:
        if pf is not None:
            side = _dominant_side(pf, ["knee", "ankle"])
            break

    lengths_px = []
    for pf in pose_frames[setup_seg.start_frame:setup_seg.end_frame + 1]:
        if pf is None:
            continue
        knee  = pf.keypoints.get(f"{side}_knee")
        ankle = pf.keypoints.get(f"{side}_ankle")
        kv    = pf.visibility.get(f"{side}_knee",  0.0)
        av    = pf.visibility.get(f"{side}_ankle", 0.0)
        if knee is not None and ankle is not None and min(kv, av) > 0.5:
            dx = knee[0] - ankle[0]
            dy = knee[1] - ankle[1]
            lengths_px.append(np.sqrt(dx ** 2 + dy ** 2))

    if not lengths_px:
        return None

    tibia_px = float(np.median(lengths_px))
    return tibia_px / tibia_m


def _check_depth_at(
    pose_frames: list[Optional[PoseFrame]],
    frame_idx: int,
) -> tuple[CriterionResult, Optional[float]]:
    """Profondità al frame di bottom. Ritorna (CriterionResult, depth_angle_deg)."""
    pf = pose_frames[frame_idx] if frame_idx < len(pose_frames) else None
    if pf is None:
        return CriterionResult(None, 0.0, "Pose non rilevata al bottom."), None

    side     = _dominant_side(pf, ["hip", "knee"])
    hip_kp   = pf.keypoints.get(f"{side}_hip")
    knee_kp  = pf.keypoints.get(f"{side}_knee")
    hip_vis  = pf.visibility.get(f"{side}_hip",  0.0)
    knee_vis = pf.visibility.get(f"{side}_knee", 0.0)

    if hip_kp is None or knee_kp is None:
        return CriterionResult(None, 0.0, "Keypoint anca/ginocchio assenti al bottom."), None

    result    = check_depth(hip_kp[0], hip_kp[1], knee_kp[0], knee_kp[1], hip_vis, knee_vis)
    dx        = hip_kp[0] - knee_kp[0]
    dy        = hip_kp[1] - knee_kp[1]
    angle_deg = float(np.degrees(np.arctan2(dy, abs(dx))))
    return result, angle_deg


def _check_lockout_at(
    pose_frames: list[Optional[PoseFrame]],
    frame_idx: int,
    label: str,
) -> CriterionResult:
    """Lockout (estensione ginocchio + anca) a un singolo frame tramite angoli geometrici."""
    pf = pose_frames[frame_idx] if frame_idx < len(pose_frames) else None
    if pf is None:
        return CriterionResult(None, 0.0, f"Pose non rilevata al lockout {label}.")

    side         = _dominant_side(pf, ["hip", "knee", "ankle"])
    shoulder_kp  = pf.keypoints.get(f"{side}_shoulder")
    hip_kp       = pf.keypoints.get(f"{side}_hip")
    knee_kp      = pf.keypoints.get(f"{side}_knee")
    ankle_kp     = pf.keypoints.get(f"{side}_ankle")
    shoulder_vis = pf.visibility.get(f"{side}_shoulder", 0.0)
    hip_vis      = pf.visibility.get(f"{side}_hip",      0.0)
    knee_vis     = pf.visibility.get(f"{side}_knee",     0.0)
    ankle_vis    = pf.visibility.get(f"{side}_ankle",    0.0)

    if None in (shoulder_kp, hip_kp, knee_kp, ankle_kp):
        return CriterionResult(None, 0.0, f"Keypoint assenti al lockout {label}.")

    return check_lockout(
        shoulder_kp, hip_kp, knee_kp, ankle_kp,
        shoulder_vis, hip_vis, knee_vis, ankle_vis,
        label,
    )


def _check_feet_series(
    pose_frames: list[Optional[PoseFrame]],
    setup_seg,
) -> CriterionResult:
    """
    Verifica contatto continuo piedi–pedana su tutta la ripetizione.
    Usa la posizione mediana della caviglia nel setup come riferimento.
    """
    # Determina lato dominante dalla prima frame valida
    side = "left"
    for pf in pose_frames:
        if pf is not None:
            side = _dominant_side(pf, ["ankle"])
            break

    ankle_key = f"{side}_ankle"
    ankle_y_all  = [
        pf.keypoints[ankle_key][1]
        if pf is not None and ankle_key in pf.keypoints else None
        for pf in pose_frames
    ]
    ankle_vis_all = [
        pf.visibility.get(ankle_key, 0.0) if pf is not None else 0.0
        for pf in pose_frames
    ]

    # Riferimento Y dal setup
    setup_range = ankle_y_all[setup_seg.start_frame : setup_seg.end_frame + 1]
    valid_setup = [v for v in setup_range if v is not None]
    if not valid_setup:
        return CriterionResult(None, 0.0, "Caviglia non rilevata nel setup.")

    initial_ankle_y = float(np.median(valid_setup))

    # Sostituisce i None con il riferimento (frame occlusi non innescano falsi positivi)
    ankle_y_filled = [y if y is not None else initial_ankle_y for y in ankle_y_all]

    return check_feet(
        ankle_y_series=ankle_y_filled,
        initial_ankle_y=initial_ankle_y,
        visibility_series=ankle_vis_all,
    )


def _aggregate_validity(criteria: list[CriterionResult]) -> tuple[bool, float]:
    """
    Calcola validità e confidenza complessiva dai criteri KO.
    - Un criterio False con confidenza >= BORDERLINE rende la rep non valida.
    - La confidenza finale è il minimo tra i criteri con esito definitivo (True/False).
      I criteri N/D (passed=None) non abbassano la confidenza: rappresentano
      "non osservabile", non "fallito".
    """
    valid = True
    confidences = []
    for c in criteria:
        if c.passed is False and c.confidence >= config.CONFIDENCE_BORDERLINE:
            valid = False
        if c.passed is not None:
            confidences.append(c.confidence)
    confidence = float(min(confidences)) if confidences else 0.0
    return valid, confidence


def _print_rejection(reason: str, bar_weight_kg: Optional[float] = None) -> None:
    result = AnalysisResult(
        valid=False,
        confidence=0.0,
        depth_ok=None,
        depth_angle_deg=None,
        initial_lockout_ok=None,
        final_lockout_ok=None,
        feet_ok=None,
        rom_m=None,
        avg_concentric_ms=None,
        peak_concentric_ms=None,
        bar_deviation_cm=None,
        bar_weight_kg=bar_weight_kg,
        estimated_1rm_pct=None,
        bar_stability_ok=None,
        symmetry_pct=None,
        symmetry_confidence=None,
        rejection_reason=reason,
    )
    print_report(result)


def _ask_float(prompt: str, required: bool) -> float | None:
    while True:
        raw = input(prompt).strip()
        if not raw:
            if required:
                print("  → Campo obbligatorio.")
                continue
            return None
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("  → Inserisci un numero valido.")


def _ask_height(prompt: str) -> float:
    """Chiede l'altezza in metri. Converte automaticamente se inserita in cm."""
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("  → Campo obbligatorio.")
            continue
        try:
            val = float(raw.replace(",", "."))
        except ValueError:
            print("  → Inserisci un numero valido (es. 1.75).")
            continue
        if val > 3.0:
            # Probabilmente inserito in cm
            val_m = val / 100.0
            confirm = input(f"  → Intendi {val_m:.2f} m? [invio = sì, n = reinserisci]: ").strip().lower()
            if confirm in ("", "s", "si", "sì", "y", "yes"):
                return val_m
            continue
        if val < 1.0:
            print("  → Altezza non plausibile. Inserisci in metri (es. 1.75).")
            continue
        return val


if __name__ == "__main__":
    main()
