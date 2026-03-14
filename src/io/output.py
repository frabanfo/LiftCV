"""
Formattazione e stampa del report finale a console.
Layout fedele al §7 dell'MVP.
"""

from dataclasses import dataclass
from typing import Optional

from src.config import VBT_ERROR_PCT, CONFIDENCE_HIGH


@dataclass
class AnalysisResult:
    # Validità complessiva
    valid: bool
    confidence: float           # 0.0 – 1.0

    # Criteri KO (None = non determinabile)
    depth_ok:          Optional[bool]
    depth_angle_deg:   Optional[float]
    initial_lockout_ok: Optional[bool]
    final_lockout_ok:  Optional[bool]
    feet_ok:           Optional[bool]

    # Metriche quantitative (None = non calcolate o barra occlusa)
    rom_m:             Optional[float]
    avg_concentric_ms: Optional[float]
    peak_concentric_ms: Optional[float]
    bar_deviation_cm:  Optional[float]

    # %1RM
    bar_weight_kg:     Optional[float]
    estimated_1rm_pct: Optional[float]

    # Flag informativi
    bar_stability_ok:  Optional[bool]
    symmetry_pct:      Optional[float]
    symmetry_confidence: Optional[float]

    # Motivo rifiuto (se analisi non affidabile)
    rejection_reason:  Optional[str] = None


def print_report(result: AnalysisResult) -> None:
    sep = "─" * 49

    if result.rejection_reason:
        print(f"\nANALISI NON AFFIDABILE")
        print(sep)
        print(f"Motivo: {result.rejection_reason}")
        print(sep)
        return

    if not result.valid:
        verdict = "NON VALIDA"
    elif result.confidence < CONFIDENCE_HIGH:
        verdict = f"BORDERLINE (confidenza {result.confidence:.0%})"
    else:
        verdict = "VALIDA"

    print(f"\nRISULTATO: {verdict}")
    print(sep)

    # Criteri KO
    print("Criteri KO:")
    if result.depth_angle_deg is not None:
        print(f"  Profondità:       {_fmt_bool(result.depth_ok)}  "
              f"(anca {result.depth_angle_deg:+.1f} cm sopra ginocchio)")
    else:
        print(f"  Profondità:       {_fmt_bool(result.depth_ok)}")
    print(f"  Lockout iniziale: {_fmt_bool(result.initial_lockout_ok)}")
    print(f"  Lockout finale:   {_fmt_bool(result.final_lockout_ok)}")
    print(f"  Piedi:            {_fmt_bool(result.feet_ok)}")

    # Metriche
    print("\nMETRICHE")
    print(f"  ROM:                  {_fmt_opt(result.rom_m, '{:.2f} m')}")
    print(f"  Velocità media conc.: {_fmt_opt(result.avg_concentric_ms, '{:.2f} m/s')}")
    print(f"  Velocità di picco:    {_fmt_opt(result.peak_concentric_ms, '{:.2f} m/s')}")
    print(f"  Deviazione barra:     {_fmt_opt(result.bar_deviation_cm, '{:.1f} cm')}")

    # %1RM
    print("\nSTIMA INTENSITÀ")
    if result.bar_weight_kg is None:
        print("  Peso bilanciere non inserito — stima %1RM disabilitata.")
    elif result.estimated_1rm_pct is None:
        print("  %1RM: non calcolabile (dati velocità non disponibili).")
    else:
        print(f"  Peso dichiarato: {result.bar_weight_kg:.0f} kg")
        print(f"  %1RM stimata:    {result.estimated_1rm_pct:.0f}% (±{VBT_ERROR_PCT}% — curva generica, non calibrata)")

    # Note informative
    print("\nNOTE INFORMATIVE")
    print(f"  Piedi:              contatto continuo {_fmt_check(result.feet_ok)}")
    print(f"  Stabilità barra:    {_fmt_opt_flag(result.bar_stability_ok)}")
    if result.symmetry_pct is not None:
        print(f"  Simmetria:          stimata (confidenza {result.symmetry_confidence:.0%})")

    # Limiti dichiarati
    print(f"\n{sep}")
    print("LIMITI DICHIARATI")
    print("  – Analisi da monocamera laterale.")
    print("  – Valgismo del ginocchio: non osservabile, escluso.")
    print("  – Simmetria: stima — non misura diretta.")
    print(f"  – %1RM: curva generica, errore atteso ±{VBT_ERROR_PCT}%.")
    print(sep)


# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt_bool(val: Optional[bool]) -> str:
    if val is None:
        return "N/D"
    return "✓" if val else "✗"

def _fmt_check(val: Optional[bool]) -> str:
    if val is None:
        return ""
    return "✓" if val else "✗"

def _fmt_opt(val: Optional[float], fmt: str) -> str:
    return fmt.format(val) if val is not None else "N/D"

def _fmt_opt_flag(val: Optional[bool]) -> str:
    if val is None:
        return "N/D"
    return "nella norma" if val else "oscillazioni rilevate"
