"""
Test di integrazione end-to-end della pipeline liftCV.

Legge gt/labels.json, esegue analyze_video() su ogni video con parametri fissi
(peso bilanciere e altezza da _params nel JSON), confronta l'output con le
etichette ground truth e stampa una tabella riassuntiva con accuracy per criterio.

Uso:
    python scripts/test_pipeline.py
    python scripts/test_pipeline.py --gt-dir gt/
    python scripts/test_pipeline.py --debug         # debug output per ogni video
    python scripts/test_pipeline.py --video squat1  # testa solo i video che matchano

I video con etichette null vengono saltati (non ancora annotati).
"""

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Aggiunge la root del progetto al path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze import analyze_video
from src.io.output import AnalysisResult


# -- Strutture dati ------------------------------------------------------------

@dataclass
class VideoLabel:
    filename: str
    valid: Optional[bool]
    depth_ok: Optional[bool]
    initial_lockout_ok: Optional[bool]
    final_lockout_ok: Optional[bool]
    feet_ok: Optional[bool]
    notes: str = ""


@dataclass
class VideoResult:
    filename: str
    label: VideoLabel
    result: Optional[AnalysisResult]
    error: Optional[str] = None

    @property
    def ran(self) -> bool:
        return self.result is not None and self.error is None

    def criterion_match(self, criterion: str) -> Optional[bool]:
        """
        Confronta il valore GT con quello ottenuto dalla pipeline.
        Ritorna True (PASS), False (FAIL), None (saltato -- GT è null o result è None).
        """
        gt_val = getattr(self.label, criterion)
        if gt_val is None or self.result is None:
            return None
        got_val = getattr(self.result, criterion)
        if got_val is None:
            return None  # pipeline non determinabile → non confrontabile
        return gt_val == got_val


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LiftCV -- Test pipeline integrazione")
    parser.add_argument("--gt-dir",  default="gt",  help="Directory contenente labels.json e i video (default: gt/)")
    parser.add_argument("--debug",   action="store_true", help="Abilita debug output della pipeline per ogni video")
    parser.add_argument("--verbose", action="store_true", help="Mostra output analisi (altrimenti soppresso)")
    parser.add_argument("--video",   default=None, help="Filtra su video il cui nome contiene questa stringa")
    args = parser.parse_args()

    gt_dir = ROOT / args.gt_dir
    labels_path = gt_dir / "labels.json"

    if not labels_path.exists():
        print(f"ERRORE: {labels_path} non trovato.")
        print("Creare gt/labels.json con le etichette ground truth.")
        sys.exit(1)

    with open(labels_path, encoding="utf-8") as f:
        data = json.load(f)

    params = data.get("_params", {})
    bar_weight_kg = float(params.get("bar_weight_kg", 100))
    height_m      = float(params.get("height_m", 1.75))

    # Carica etichette (ignora chiavi che iniziano con _)
    labels: list[VideoLabel] = []
    for filename, entry in data.items():
        if filename.startswith("_"):
            continue
        if args.video and args.video not in filename:
            continue
        labels.append(VideoLabel(
            filename=filename,
            valid=entry.get("valid"),
            depth_ok=entry.get("depth_ok"),
            initial_lockout_ok=entry.get("initial_lockout_ok"),
            final_lockout_ok=entry.get("final_lockout_ok"),
            feet_ok=entry.get("feet_ok"),
            notes=entry.get("notes", ""),
        ))

    if not labels:
        print("Nessun video trovato in labels.json.")
        sys.exit(0)

    # Conta quanti hanno etichette compilate
    labeled = [l for l in labels if l.valid is not None]
    print(f"\nLiftCV -- Test Pipeline")
    print(f"{'-' * 60}")
    print(f"GT dir:        {gt_dir}")
    print(f"Video totali:  {len(labels)}  |  Annotati: {len(labeled)}  |  Da annotare: {len(labels) - len(labeled)}")
    print(f"Parametri:     {bar_weight_kg:.0f} kg  x  {height_m:.2f} m")
    print(f"{'-' * 60}\n")

    # -- Esegui pipeline su ogni video -----------------------------------------
    results: list[VideoResult] = []
    for label in labels:
        video_path = gt_dir / label.filename
        prefix = f"[{label.filename}]"

        if not video_path.exists():
            print(f"{prefix}  FILE NON TROVATO -- saltato")
            results.append(VideoResult(filename=label.filename, label=label, result=None, error="file not found"))
            continue

        if label.valid is None:
            print(f"{prefix}  etichetta non compilata -- saltato")
            results.append(VideoResult(filename=label.filename, label=label, result=None, error="not labeled"))
            continue

        print(f"{prefix}  analisi in corso...", end="", flush=True)
        try:
            if args.verbose or args.debug:
                result = analyze_video(str(video_path), bar_weight_kg, height_m, debug=args.debug)
            else:
                with redirect_stdout(io.StringIO()):
                    result = analyze_video(str(video_path), bar_weight_kg, height_m, debug=False)
            print(" OK")
            results.append(VideoResult(filename=label.filename, label=label, result=result))
        except Exception as e:
            print(f" ERRORE: {e}")
            results.append(VideoResult(filename=label.filename, label=label, result=None, error=str(e)))

    # -- Tabella risultati -----------------------------------------------------
    criteria = ["valid", "depth_ok", "initial_lockout_ok", "final_lockout_ok", "feet_ok"]
    col_w = 8  # width per colonna criterio

    header_labels = ["valid", "depth", "init_lo", "fin_lo", "feet"]
    header = f"{'Video':<18}" + "".join(f"{h:^{col_w}}" for h in header_labels) + "  ESITO"
    sep    = "-" * len(header)

    print(f"\nRISULTATI TEST")
    print(sep)
    print(header)
    print(sep)

    ran_results = [r for r in results if r.ran]
    for r in results:
        name = r.filename[:17]
        if not r.ran:
            status = r.error or "skip"
            print(f"{name:<18}{'--':^{col_w * len(criteria)}}  [{status}]")
            continue

        cells = []
        all_pass = True
        for crit in criteria:
            match = r.criterion_match(crit)
            if match is True:
                cells.append(f"{'PASS':^{col_w}}")
            elif match is False:
                cells.append(f"{'FAIL':^{col_w}}")
                all_pass = False
            else:
                cells.append(f"{'N/D':^{col_w}}")

        esito = "PASS" if all_pass else "FAIL"
        print(f"{name:<18}" + "".join(cells) + f"  {esito}")

    print(sep)

    # -- Accuracy per criterio -------------------------------------------------
    if ran_results:
        print(f"\nACCURACY  ({len(ran_results)} video analizzati)")
        for crit, label_name in zip(criteria, header_labels):
            matches = [r.criterion_match(crit) for r in ran_results]
            total   = sum(1 for m in matches if m is not None)
            passed  = sum(1 for m in matches if m is True)
            nd_count = sum(1 for m in matches if m is None)
            if total > 0:
                pct = passed / total * 100
                nd_note = f"  ({nd_count} N/D)" if nd_count else ""
                print(f"  {label_name:<12} {passed}/{total}  ({pct:.0f}%){nd_note}")
            else:
                print(f"  {label_name:<12} N/D (nessun confronto possibile)")

        # Overall pass rate
        overall = [r for r in ran_results if all(
            r.criterion_match(c) is not False for c in criteria
        )]
        print(f"\n  OVERALL  {len(overall)}/{len(ran_results)} video con tutti i criteri PASS o N/D")
    else:
        print("\nNessun video eseguito -- annotare gt/labels.json per iniziare i test.")

    print(f"{'-' * 60}\n")

    # Exit code: 0 se tutti i video annotati passano, 1 altrimenti
    any_fail = any(
        r.criterion_match(c) is False
        for r in ran_results
        for c in criteria
    )
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
