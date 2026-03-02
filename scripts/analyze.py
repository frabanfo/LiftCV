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

# Aggiunge la root del progetto al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io.video import load_video, iter_frames
from src.io.output import AnalysisResult, print_report
from src.tracking.pose import PoseEstimator
from src.tracking.bar_tracker import BarTracker, detect_bar_in_roi
from src.segmentation import segment_repetition
from src.validation import check_depth, check_lockout, check_feet
from src.metrics import compute_metrics
from src import config


def main():
    parser = argparse.ArgumentParser(description="LiftCV — Analisi squat da video laterale")
    parser.add_argument("video", help="Percorso del video da analizzare")
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
    bar_weight_kg = _ask_float("Peso sul bilanciere (kg) [obbligatorio]: ", required=True)
    body_weight_kg = _ask_float("Peso corporeo (kg) [invio per saltare]: ", required=False)
    historical_1rm_kg = _ask_float("1RM storico (kg) [invio per saltare]: ", required=False)

    # ── TODO: pipeline completa ──────────────────────────────────────────────
    # Placeholder — la pipeline verrà implementata incrementalmente.
    # Al momento stampa un report di esempio per verificare che tutto si avvii.
    print("\n[Pipeline non ancora implementata — report di esempio]\n")

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
        rejection_reason="Pipeline non ancora implementata.",
    )
    print_report(result)

    cap.release()


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


if __name__ == "__main__":
    main()
