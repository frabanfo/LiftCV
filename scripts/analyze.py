"""
Entry point MVP -- LiftCV.

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


def analyze_video(
    video_path: str,
    bar_weight_kg: float,
    height_m: float,
    debug: bool = False,
) -> AnalysisResult:
    """
    Esegue la pipeline completa di analisi squat.

    Parametri:
        video_path:    Percorso al video da analizzare.
        bar_weight_kg: Peso sul bilanciere in kg (obbligatorio).
        height_m:      Altezza atleta in metri (obbligatorio).
        debug:         Se True, stampa diagnostica aggiuntiva a console.

    Ritorna:
        AnalysisResult per tutti i casi (inclusi failure/rejection).
        Se il video non si apre, lancia FileNotFoundError o ValueError.
    """
    # -- Caricamento video ----------------------------------------------------
    cap, meta = load_video(video_path)

    print(f"\nVideo caricato: {meta.path.name}")
    print(f"  {meta.frame_count} frame  |  {meta.fps:.1f} fps  |  {meta.width}x{meta.height}px")
    print("\nAnalisi in corso...")

    # -- Scan frame: pose + barra ----------------------------------------------
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

    if debug:
        pose_detected = sum(1 for pf in pose_frames if pf is not None)
        print(f"\n[DEBUG] Pose rilevata:   {pose_detected}/{total_frames} frame ({pose_detected/total_frames:.0%})")
        print(f"[DEBUG] Barra rilevata:  {detection_count}/{total_frames} frame ({detection_count/total_frames:.0%})")

    if not bar_initialized:
        return _make_rejection_result(
            "Barra non rilevata nel video. "
            "Assicurarsi che la barra sia visibile e illuminata correttamente.",
            bar_weight_kg,
        )

    # -- Segmentazione ---------------------------------------------------------
    bar_y_arr = np.array([y if y is not None else np.nan for y in bar_y_px])

    if debug:
        valid_y = bar_y_arr[~np.isnan(bar_y_arr)]
        bottom_idx_dbg = int(np.nanargmax(bar_y_arr))
        print(f"\n[DEBUG] Segnale barra Y:")
        print(f"  Range:      {valid_y.min():.0f}-{valid_y.max():.0f} px  (delta={valid_y.max()-valid_y.min():.0f} px)")
        print(f"  NaN:        {np.isnan(bar_y_arr).sum()}/{len(bar_y_arr)} frame")
        print(f"  Bottom idx: frame {bottom_idx_dbg}  (Y={bar_y_arr[bottom_idx_dbg]:.0f} px)")
        print(f"  Inizio:     Y={valid_y[0]:.0f} px  |  Fine: Y={valid_y[-1]:.0f} px")

    seg = segment_repetition(bar_y_arr, meta.fps)

    if debug:
        status = "OK" if seg.success else "FALLITA"
        print(f"\n[DEBUG] Segmentazione -- {status}")
        for s in seg.segments:
            flag = "  <- PROBLEMA" if s.confidence < CONFIDENCE_BORDERLINE else ""
            print(f"  {s.phase.name:<10} frame {s.start_frame:>4}-{s.end_frame:<4}  conf={s.confidence:.0%}{flag}")
        if not seg.success:
            print(f"  Motivo: {seg.failure_reason}")

    if not seg.success:
        return _make_rejection_result(seg.failure_reason, bar_weight_kg)

    bottom_seg  = seg.get(Phase.BOTTOM)
    setup_seg   = seg.get(Phase.SETUP)
    lockout_seg = seg.get(Phase.LOCKOUT)
    descent_seg = seg.get(Phase.DESCENT)
    ascent_seg  = seg.get(Phase.ASCENT)

    # -- Calibrazione scala ----------------------------------------------------
    px_per_meter = _compute_px_per_meter(pose_frames, setup_seg, height_m)
    if px_per_meter is None:
        print("  Avviso: scala non calcolabile (tibia non visibile nel setup) -- metriche in m/s = N/D.")

    # -- Validazione criteri KO ------------------------------------------------
    depth_result, depth_display = _check_depth_at(
        pose_frames,
        bottom_seg.start_frame,
        search_start=descent_seg.start_frame,
        search_end=ascent_seg.end_frame,
        px_per_meter=px_per_meter,
    )
    init_lockout  = _check_lockout_at(
        pose_frames, setup_seg.end_frame, "iniziale",
        search_start=setup_seg.start_frame,
        search_end=setup_seg.end_frame,
    )
    final_lockout = _check_lockout_at(
        pose_frames, lockout_seg.start_frame, "finale",
        search_start=lockout_seg.start_frame,
        search_end=lockout_seg.end_frame,
    )
    feet_result   = _check_feet_series(pose_frames, setup_seg, descent_seg.start_frame, ascent_seg.end_frame, px_per_meter=px_per_meter)

    if debug:
        _debug_depth(pose_frames, bottom_seg.start_frame, descent_seg.start_frame, ascent_seg.end_frame)
        _debug_feet(pose_frames, setup_seg, descent_seg.start_frame, ascent_seg.end_frame)

    # -- Metriche -------------------------------------------------------------
    metrics = compute_metrics(
        bar_y_px=bar_y_px,
        bar_x_px=bar_x_px,
        segmentation=seg,
        fps=meta.fps,
        px_per_meter=px_per_meter,
        bar_weight_kg=bar_weight_kg,
    )

    # -- Validità e confidenza complessiva -------------------------------------
    ko_criteria = [depth_result, init_lockout, final_lockout, feet_result]
    valid, confidence = _aggregate_validity(ko_criteria)

    bar_stability_ok: Optional[bool] = (
        metrics.bar_deviation_cm < 5.0
        if metrics.bar_deviation_cm is not None else None
    )

    return AnalysisResult(
        valid=valid,
        confidence=confidence,
        depth_ok=depth_result.passed,
        depth_angle_deg=depth_display,
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


def main():
    parser = argparse.ArgumentParser(description="LiftCV -- Analisi squat da video laterale")
    parser.add_argument("video", help="Percorso del video da analizzare")
    parser.add_argument("--debug", action="store_true", help="Stampa info diagnostiche sul segnale barra e segmentazione")
    args = parser.parse_args()

    # -- Caricamento video (validazione path) ---------------------------------
    # Fatto qui per dare feedback immediato prima dell'input interattivo
    try:
        from src.io.video import load_video as _check_video
        cap_check, _ = _check_video(args.video)
        cap_check.release()
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERRORE INPUT: {e}")
        sys.exit(1)

    # -- Dati utente ----------------------------------------------------------
    bar_weight_kg     = _ask_float("Peso sul bilanciere (kg) [obbligatorio]: ", required=True)
    height_m          = _ask_height("Altezza atleta (m, es. 1.75) [obbligatorio]: ")
    _ask_float("Peso corporeo (kg) [invio per saltare]: ", required=False)    # noqa: F841
    _ask_float("1RM storico (kg) [invio per saltare]: ",   required=False)    # noqa: F841

    result = analyze_video(args.video, bar_weight_kg, height_m, debug=args.debug)
    print_report(result)


# -- Pipeline helpers ----------------------------------------------------------

def _detect_bar(
    pf: Optional[PoseFrame],
    frame_bgr,
    width: int,
    height: int,
) -> Optional[tuple[float, float]]:
    """
    Stima la posizione della barra nel frame corrente.

    Strategia per ripresa laterale back-squat:
    Usa la spalla del lato dominante (il lato vicino alla telecamera) come proxy.
    La spalla è sempre visibile in ripresa laterale e la sua Y segue la barra con
    offset costante -> corretto per segmentazione, ROM, velocità (tutti calcoli delta).

    Il color detection cromatico precedente è stato rimosso: con una ripresa
    laterale tipica produceva <50% di rilevazioni valide e 789px di range spurio
    a causa di match su sfondo/abbigliamento.
    """
    if pf is None:
        return None

    side = _dominant_side(pf, ["shoulder", "hip"])
    shoulder_kp  = pf.keypoints.get(f"{side}_shoulder")
    shoulder_vis = pf.visibility.get(f"{side}_shoulder", 0.0)

    if shoulder_kp is None or shoulder_vis < 0.5:
        return None

    return (shoulder_kp[0], shoulder_kp[1])


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
    Lunghezza tibia stimata = altezza x TIBIA_HEIGHT_RATIO (Drillis & Contini).
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
    search_start: int,
    search_end: int,
    px_per_meter: Optional[float] = None,
) -> tuple[CriterionResult, Optional[float]]:
    """
    Profondità sulla finestra [search_start, search_end] (tutta la fase discesa->risalita).
    Tra i frame con confidenza >= BORDERLINE, sceglie quello con massima profondità
    (hip_y - knee_y più grande = anca più bassa rispetto al ginocchio).

    Ritorna (CriterionResult, depth_cm) dove depth_cm è la distanza verticale
    anca-ginocchio in cm (positivo = anca sopra ginocchio) al frame più profondo.
    Se px_per_meter non è disponibile, ritorna l'angolo in gradi al posto di depth_cm.

    Fallback A: se nessun frame supera la soglia di confidenza, prende il frame con
    massima profondità (max dy) -- non quello con massima confidenza.
    Fallback B: se non ci sono candidati, restituisce N/D.
    """
    candidates: list[tuple[float, float, CriterionResult]] = []  # (dy, conf, result)

    for fi in range(max(0, search_start), min(len(pose_frames), search_end + 1)):
        pf = pose_frames[fi]
        if pf is None:
            continue
        side     = _dominant_side(pf, ["hip", "knee"])
        hip_kp   = pf.keypoints.get(f"{side}_hip")
        knee_kp  = pf.keypoints.get(f"{side}_knee")
        hip_vis  = pf.visibility.get(f"{side}_hip",  0.0)
        knee_vis = pf.visibility.get(f"{side}_knee", 0.0)
        if hip_kp is None or knee_kp is None:
            continue
        result = check_depth(
            hip_kp[0], hip_kp[1], knee_kp[0], knee_kp[1],
            hip_vis, knee_vis, px_per_meter,
        )
        dy = hip_kp[1] - knee_kp[1]
        candidates.append((dy, min(hip_vis, knee_vis), result))

    if not candidates:
        return CriterionResult(None, 0.0, "Pose non rilevata al bottom."), None

    # Tra i frame con confidence sufficiente, prendi quello più profondo (max dy).
    adequate = [(dy, conf, res) for dy, conf, res in candidates if conf >= CONFIDENCE_BORDERLINE]
    if adequate:
        best_dy, _, best_result = max(adequate, key=lambda x: x[0])
    else:
        # Fallback: nessun frame supera la soglia di confidenza.
        # Prendi il frame più profondo tra tutti -- non quello più visibile.
        best_dy, _, best_result = max(candidates, key=lambda x: x[0])
        best_result = CriterionResult(
            passed=None,
            confidence=0.0,
            detail=f"Visibilità anca/ginocchio insufficiente al bottom.",
        )

    # Valore di ritorno per il report: cm (se calibrato) o angolo (fallback)
    if px_per_meter is not None:
        display_val = -best_dy / px_per_meter * 100.0  # cm, positivo = anca sopra ginocchio
    else:
        dx = 0.0  # non abbiamo il dx del best_dy; usiamo None
        display_val = None

    return best_result, display_val


def _check_lockout_at(
    pose_frames: list[Optional[PoseFrame]],
    frame_idx: int,
    label: str,
    search_start: Optional[int] = None,
    search_end: Optional[int] = None,
    window: int = 5,
) -> CriterionResult:
    """
    Lockout (estensione ginocchio + anca) in un range di frame.
    Se search_start/search_end sono forniti, scansiona l'intera fase;
    altrimenti usa una finestra +/-window intorno a frame_idx.
    Sceglie il frame con la migliore estensione tra quelli validi.
    """
    best_result: Optional[CriterionResult] = None
    best_extension: float = -1.0

    start = search_start if search_start is not None else frame_idx - window
    end   = search_end   if search_end   is not None else frame_idx + window

    for fi in range(
        max(0, start),
        min(len(pose_frames), end + 1),
    ):
        pf = pose_frames[fi]
        if pf is None:
            continue
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
            continue

        result = check_lockout(
            shoulder_kp, hip_kp, knee_kp, ankle_kp,
            shoulder_vis, hip_vis, knee_vis, ankle_vis,
            label,
        )
        # "extension quality" = min(knee_angle, hip_angle) proxy via confidencexpassed
        # Use confidence as a secondary sort key; primary: prefer passed=True > None > False
        extension_score = (
            (2.0 if result.passed is True else (1.0 if result.passed is None else 0.0))
            + result.confidence
        )
        if extension_score > best_extension:
            best_extension = extension_score
            best_result    = result

    if best_result is None:
        return CriterionResult(None, 0.0, f"Pose non rilevata al lockout {label}.")
    return best_result


def _check_feet_series(
    pose_frames: list[Optional[PoseFrame]],
    setup_seg,
    rep_start: int,
    rep_end: int,
    px_per_meter: Optional[float] = None,
) -> CriterionResult:
    """
    Verifica che i piedi non si sollevino durante la fase di movimento
    (dalla discesa alla fine dell'ascesa). Non include walkout e re-rack.

    Riferimento Y: mediana del tallone sull'intera fase di setup (non solo
    gli ultimi 5 frame), per robustezza quando la pose non è rilevata in
    alcuni frame iniziali.
    """
    side = "left"
    for pf in pose_frames:
        if pf is not None:
            side = _dominant_side(pf, ["heel"])
            break

    heel_key = f"{side}_heel"
    heel_y_all = [
        pf.keypoints[heel_key][1]
        if pf is not None and heel_key in pf.keypoints else None
        for pf in pose_frames
    ]
    heel_vis_all = [
        pf.visibility.get(heel_key, 0.0) if pf is not None else 0.0
        for pf in pose_frames
    ]

    # Riferimento Y: mediana sull'intera fase di setup (atleta in stance).
    # Fallback: primi frame della discesa se il setup è privo di rilevazioni.
    valid_ref = [
        v for v in heel_y_all[setup_seg.start_frame : setup_seg.end_frame + 1]
        if v is not None
    ]
    if not valid_ref:
        fallback_end = min(len(heel_y_all), rep_start + 5)
        valid_ref = [v for v in heel_y_all[rep_start:fallback_end] if v is not None]
    if not valid_ref:
        return CriterionResult(None, 0.0, "Tallone non rilevato nel setup.")

    initial_heel_y = float(np.median(valid_ref))

    heel_y_rep    = heel_y_all[rep_start : rep_end + 1]
    heel_vis_rep  = heel_vis_all[rep_start : rep_end + 1]
    heel_y_filled = [y if y is not None else initial_heel_y for y in heel_y_rep]

    return check_feet(
        ankle_y_series=heel_y_filled,
        initial_ankle_y=initial_heel_y,
        visibility_series=heel_vis_rep,
        px_per_meter=px_per_meter,
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


def _make_rejection_result(reason: str, bar_weight_kg: Optional[float] = None) -> AnalysisResult:
    """Crea un AnalysisResult di rigetto con motivo esplicito."""
    return AnalysisResult(
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


def _debug_depth(
    pose_frames: list[Optional[PoseFrame]],
    bottom_frame: int,
    search_start: int,
    search_end: int,
) -> None:
    """Stampa i landmark anca/ginocchio in ogni frame della finestra di depth check."""
    print(f"\n[DEBUG] Depth check -- finestra frame {search_start}-{search_end}  (bottom={bottom_frame}):")
    for fi in range(max(0, search_start), min(len(pose_frames), search_end + 1)):
        pf = pose_frames[fi]
        if pf is None:
            print(f"  frame {fi:3d}  pose=None")
            continue
        side = _dominant_side(pf, ["hip", "knee"])
        hip_kp  = pf.keypoints.get(f"{side}_hip")
        knee_kp = pf.keypoints.get(f"{side}_knee")
        hv = pf.visibility.get(f"{side}_hip",  0.0)
        kv = pf.visibility.get(f"{side}_knee", 0.0)
        if hip_kp and knee_kp:
            dy = hip_kp[1] - knee_kp[1]
            dx = hip_kp[0] - knee_kp[0]
            ang = float(np.degrees(np.arctan2(dy, abs(dx))))
            marker = " <- bottom" if fi == bottom_frame else ""
            print(
                f"  frame {fi:3d}  side={side}"
                f"  hip=({hip_kp[0]:.0f},{hip_kp[1]:.0f}) vis={hv:.2f}"
                f"  knee=({knee_kp[0]:.0f},{knee_kp[1]:.0f}) vis={kv:.2f}"
                f"  dy={dy:+.0f}  angle={ang:+.1f} deg{marker}"
            )
        else:
            print(f"  frame {fi:3d}  side={side}  keypoint mancante")


def _debug_feet(
    pose_frames: list[Optional[PoseFrame]],
    setup_seg,
    rep_start: int,
    rep_end: int,
) -> None:
    """Stampa lo spostamento del tallone frame per frame durante la fase di movimento."""
    side = "left"
    for pf in pose_frames:
        if pf is not None:
            side = _dominant_side(pf, ["heel"])
            break

    heel_key = f"{side}_heel"
    heel_y_all = [
        pf.keypoints[heel_key][1] if pf is not None and heel_key in pf.keypoints else None
        for pf in pose_frames
    ]
    heel_vis_all = [
        pf.visibility.get(heel_key, 0.0) if pf is not None else 0.0
        for pf in pose_frames
    ]

    # Mirror the same reference logic as _check_feet_series
    valid_ref = [v for v in heel_y_all[setup_seg.start_frame : setup_seg.end_frame + 1] if v is not None]
    if not valid_ref:
        fallback_end = min(len(heel_y_all), rep_start + 5)
        valid_ref = [v for v in heel_y_all[rep_start:fallback_end] if v is not None]
    ref_y = float(np.median(valid_ref)) if valid_ref else None

    if ref_y is None:
        print(f"\n[DEBUG] Feet (heel {side}) -- riferimento non disponibile (frame {rep_start}-{rep_end})")
        return

    lift_thresh_px = 30  # approx; exact value depends on px_per_meter
    print(f"\n[DEBUG] Feet (heel {side}) -- ref_y={ref_y:.0f}px  frame {rep_start}-{rep_end}  (soglia ~{lift_thresh_px}px):")
    for fi in range(rep_start, rep_end + 1):
        y   = heel_y_all[fi]
        vis = heel_vis_all[fi]
        if y is None:
            delta = "N/D"
            flag  = ""
        else:
            d = ref_y - y          # positivo = tallone salito
            delta = f"{d:+.0f}px"
            flag  = "  <- LIFT" if d > lift_thresh_px and vis >= CONFIDENCE_BORDERLINE else ""
        print(f"  frame {fi:3d}  heel_y={str(round(y)) if y else 'N/D':>6}  delta={delta:>8}  vis={vis:.2f}{flag}")


def _ask_float(prompt: str, required: bool) -> float | None:
    while True:
        raw = input(prompt).strip()
        if not raw:
            if required:
                print("  -> Campo obbligatorio.")
                continue
            return None
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            print("  -> Inserisci un numero valido.")


def _ask_height(prompt: str) -> float:
    """Chiede l'altezza in metri. Converte automaticamente se inserita in cm."""
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("  -> Campo obbligatorio.")
            continue
        try:
            val = float(raw.replace(",", "."))
        except ValueError:
            print("  -> Inserisci un numero valido (es. 1.75).")
            continue
        if val > 3.0:
            # Probabilmente inserito in cm
            val_m = val / 100.0
            confirm = input(f"  -> Intendi {val_m:.2f} m? [invio = sì, n = reinserisci]: ").strip().lower()
            if confirm in ("", "s", "si", "sì", "y", "yes"):
                return val_m
            continue
        if val < 1.0:
            print("  -> Altezza non plausibile. Inserisci in metri (es. 1.75).")
            continue
        return val


if __name__ == "__main__":
    main()
