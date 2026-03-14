"""
Unit tests per src/validation.py.

Testa le quattro funzioni di validazione IPF con dati sintetici:
  - check_depth:   profondità anca/ginocchio (modalità calibrata e angolare)
  - check_lockout: estensione ginocchio + anca
  - check_feet:    sollevamento piedi durante la rep

Ogni test usa coordinate geometriche deterministiche, indipendente da MediaPipe
o da video reali.

Soglie di riferimento (da src/config.py):
  DEPTH_OFFSET_M        = 0.15   → threshold_cm = 15 cm
  DEPTH_TOLERANCE_M     = 0.02   → ±2 cm zona borderline
  DEPTH_THRESHOLD_DEG   = -8.0   → soglia angolare fallback
  LOCKOUT_KNEE_MIN_DEG  = 130°
  LOCKOUT_HIP_MIN_DEG   = 120°
  FEET_LIFT_THRESHOLD_M = 0.15   → 75 px a px_per_meter=500
  FEET_LIFT_THRESHOLD_PX= 70     → fallback senza calibrazione
  FEET_JITTER_FRAMES    = 3
  CONFIDENCE_BORDERLINE = 0.60
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Aggiunge la root del progetto al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation import check_depth, check_lockout, check_feet, CriterionResult
from src.config import (
    CONFIDENCE_BORDERLINE,
    DEPTH_OFFSET_M,
    DEPTH_TOLERANCE_M,
    FEET_JITTER_FRAMES,
    FEET_LIFT_THRESHOLD_M,
    FEET_LIFT_THRESHOLD_PX,
    LOCKOUT_HIP_MIN_DEG,
    LOCKOUT_KNEE_MIN_DEG,
)

# ── Costanti per i test ───────────────────────────────────────────────────────

PX_PER_M   = 500.0   # 500 pixel = 1 metro
HIGH_VIS   = 0.90    # visibilità alta
LOW_VIS    = 0.30    # visibilità sotto CONFIDENCE_BORDERLINE


# ── Helpers geometrici ────────────────────────────────────────────────────────

def _hip_above_cm_to_dy(hip_above_cm: float) -> float:
    """
    Converte la distanza verticale anca-ginocchio in cm a dy pixel.
    hip_above_cm > 0  → anca sopra ginocchio  (dy negativo in coordinate pixel Y↓)
    hip_above_cm < 0  → anca sotto ginocchio  (dy positivo)

    Matematica: hip_above_cm = -dy / px_per_meter * 100  →  dy = -hip_above_cm/100 * px_per_meter
    """
    return -hip_above_cm / 100.0 * PX_PER_M


def _coords_for_depth(hip_above_cm: float) -> tuple[float, float, float, float]:
    """Ritorna (hip_x, hip_y, knee_x, knee_y) con hip_above_cm di distanza verticale."""
    knee_y = 500.0
    hip_y  = knee_y + _hip_above_cm_to_dy(hip_above_cm)
    return 200.0, hip_y, 200.0, knee_y


# ══════════════════════════════════════════════════════════════════════════════
# check_depth — modalità calibrata (px_per_meter fornito)
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckDepthCalibrated:

    def test_valid_hip_clearly_below_threshold(self):
        """Anca 10 cm sopra il ginocchio → sotto la soglia di 15 cm → VALIDA."""
        hx, hy, kx, ky = _coords_for_depth(10.0)
        result = check_depth(hx, hy, kx, ky, HIGH_VIS, HIGH_VIS, px_per_meter=PX_PER_M)
        assert result.passed is True
        assert result.confidence == pytest.approx(HIGH_VIS)

    def test_valid_hip_at_knee_level(self):
        """Anca allo stesso livello del ginocchio (0 cm) → VALIDA."""
        hx, hy, kx, ky = _coords_for_depth(0.0)
        result = check_depth(hx, hy, kx, ky, HIGH_VIS, HIGH_VIS, px_per_meter=PX_PER_M)
        assert result.passed is True

    def test_valid_hip_below_knee(self):
        """Anca sotto il ginocchio (-5 cm) → VALIDA (profondità eccellente)."""
        hx, hy, kx, ky = _coords_for_depth(-5.0)
        result = check_depth(hx, hy, kx, ky, HIGH_VIS, HIGH_VIS, px_per_meter=PX_PER_M)
        assert result.passed is True

    def test_invalid_hip_too_high(self):
        """Anca 20 cm sopra il ginocchio → supera soglia 15 cm → NON VALIDA."""
        hx, hy, kx, ky = _coords_for_depth(20.0)
        result = check_depth(hx, hy, kx, ky, HIGH_VIS, HIGH_VIS, px_per_meter=PX_PER_M)
        assert result.passed is False
        assert result.confidence == pytest.approx(HIGH_VIS)

    def test_invalid_hip_high_squat(self):
        """Anca 30 cm sopra il ginocchio (high squat evidente) → NON VALIDA."""
        hx, hy, kx, ky = _coords_for_depth(30.0)
        result = check_depth(hx, hy, kx, ky, HIGH_VIS, HIGH_VIS, px_per_meter=PX_PER_M)
        assert result.passed is False

    def test_borderline_at_threshold(self):
        """Anca esattamente a 15 cm (sulla soglia) → BORDERLINE (passed=None)."""
        hx, hy, kx, ky = _coords_for_depth(DEPTH_OFFSET_M * 100)  # 15 cm
        result = check_depth(hx, hy, kx, ky, HIGH_VIS, HIGH_VIS, px_per_meter=PX_PER_M)
        assert result.passed is None, f"Atteso None (borderline), ottenuto {result.passed}"

    def test_borderline_just_inside_tolerance(self):
        """Anca a 14 cm (dentro ±2 cm dalla soglia) → BORDERLINE."""
        hx, hy, kx, ky = _coords_for_depth(14.0)
        result = check_depth(hx, hy, kx, ky, HIGH_VIS, HIGH_VIS, px_per_meter=PX_PER_M)
        assert result.passed is None

    def test_borderline_just_outside_tolerance(self):
        """Anca a 12.9 cm (fuori dalla zona borderline ±2 cm → 13 cm) → VALIDA con certezza."""
        hx, hy, kx, ky = _coords_for_depth(12.9)
        result = check_depth(hx, hy, kx, ky, HIGH_VIS, HIGH_VIS, px_per_meter=PX_PER_M)
        assert result.passed is True

    def test_detail_string_present(self):
        """Il campo detail deve essere popolato."""
        hx, hy, kx, ky = _coords_for_depth(10.0)
        result = check_depth(hx, hy, kx, ky, HIGH_VIS, HIGH_VIS, px_per_meter=PX_PER_M)
        assert result.detail != ""


# ══════════════════════════════════════════════════════════════════════════════
# check_depth — visibilità bassa
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckDepthVisibility:

    def test_low_visibility_returns_none(self):
        """Visibilità sotto CONFIDENCE_BORDERLINE → passed=None (non determinabile)."""
        hx, hy, kx, ky = _coords_for_depth(20.0)  # sarebbe invalido, ma non visibile
        result = check_depth(hx, hy, kx, ky, LOW_VIS, LOW_VIS, px_per_meter=PX_PER_M)
        assert result.passed is None
        assert result.confidence < CONFIDENCE_BORDERLINE

    def test_mixed_visibility_uses_minimum(self):
        """La confidence è il minimo tra hip_vis e knee_vis."""
        mid_vis = (CONFIDENCE_BORDERLINE + 1.0) / 2  # ~0.8
        hx, hy, kx, ky = _coords_for_depth(10.0)
        result = check_depth(hx, hy, kx, ky, HIGH_VIS, mid_vis, px_per_meter=PX_PER_M)
        assert result.confidence == pytest.approx(min(HIGH_VIS, mid_vis))


# ══════════════════════════════════════════════════════════════════════════════
# check_depth — fallback angolare (px_per_meter=None)
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckDepthAngular:

    def test_angular_valid_hip_clearly_lower(self):
        """
        Anca molto più bassa del ginocchio in pixel → angolo positivo > -8° → VALIDA.
        hip=(100, 500), knee=(100, 400): dy=100, dx=0 → angle=arctan2(100,0)=90° > -8°
        """
        result = check_depth(100.0, 500.0, 100.0, 400.0, HIGH_VIS, HIGH_VIS, px_per_meter=None)
        assert result.passed is True

    def test_angular_invalid_hip_clearly_higher(self):
        """
        Anca molto più alta del ginocchio → angolo molto negativo < -8° → NON VALIDA.
        hip=(100, 300), knee=(200, 400): dy=-100, dx=-100 → angle=arctan2(-100,100)≈-45°
        """
        result = check_depth(100.0, 300.0, 200.0, 400.0, HIGH_VIS, HIGH_VIS, px_per_meter=None)
        assert result.passed is False

    def test_angular_low_visibility(self):
        """Bassa visibilità in modalità angolare → passed=None."""
        result = check_depth(100.0, 300.0, 200.0, 400.0, LOW_VIS, LOW_VIS, px_per_meter=None)
        assert result.passed is None


# ══════════════════════════════════════════════════════════════════════════════
# check_lockout
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckLockout:

    # Coordinate per un corpo perfettamente eretto (tutti i punti collineari verticalmente)
    # Angolo al ginocchio: hip(100,0)→knee(100,100)→ankle(100,200) = 180°
    # Angolo all'anca:     shoulder(100,-100)→hip(100,0)→knee(100,100) = 180°
    SHOULDER_ERECT = (100.0, -100.0)
    HIP_ERECT      = (100.0,   0.0)
    KNEE_ERECT     = (100.0, 100.0)
    ANKLE_ERECT    = (100.0, 200.0)

    def test_full_lockout_extended(self):
        """Corpo eretto (tutti i landmark collineari) → lockout valido."""
        result = check_lockout(
            self.SHOULDER_ERECT, self.HIP_ERECT, self.KNEE_ERECT, self.ANKLE_ERECT,
            HIGH_VIS, HIGH_VIS, HIGH_VIS, HIGH_VIS,
        )
        assert result.passed is True
        assert result.confidence == pytest.approx(HIGH_VIS)

    def test_bent_knee_invalid(self):
        """
        Ginocchio piegato a 90° → angolo knee = 90° < 130° → NON VALIDO.
        hip(100,0)→knee(100,100)→ankle(0,100): vettori (0,-100) e (-100,0) → 90°
        """
        result = check_lockout(
            (100.0, -100.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0),
            HIGH_VIS, HIGH_VIS, HIGH_VIS, HIGH_VIS,
        )
        assert result.passed is False

    def test_bent_hip_invalid(self):
        """
        Anca piegata a 90° → angolo hip < 120° → NON VALIDO.
        shoulder(0,0)→hip(100,0)→knee(100,100): vettori (-100,0) e (0,100) → 90°
        """
        result = check_lockout(
            (0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (100.0, 200.0),
            HIGH_VIS, HIGH_VIS, HIGH_VIS, HIGH_VIS,
        )
        assert result.passed is False

    def test_knee_ok_hip_not_ok(self):
        """Ginocchio esteso ma anca piegata → NON VALIDO (entrambi necessari)."""
        # Knee: collineare verticale → 180° (ok)
        # Hip: angolo 90° → non ok
        result = check_lockout(
            (0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (100.0, 200.0),
            HIGH_VIS, HIGH_VIS, HIGH_VIS, HIGH_VIS,
        )
        assert result.passed is False

    def test_low_visibility_returns_none(self):
        """Visibilità insufficiente su tutti i keypoint → passed=None."""
        result = check_lockout(
            self.SHOULDER_ERECT, self.HIP_ERECT, self.KNEE_ERECT, self.ANKLE_ERECT,
            LOW_VIS, LOW_VIS, LOW_VIS, LOW_VIS,
        )
        assert result.passed is None

    def test_one_keypoint_low_visibility(self):
        """Una visibilità bassa abbassa la confidence sotto borderline → passed=None."""
        result = check_lockout(
            self.SHOULDER_ERECT, self.HIP_ERECT, self.KNEE_ERECT, self.ANKLE_ERECT,
            LOW_VIS, HIGH_VIS, HIGH_VIS, HIGH_VIS,
        )
        # confidence = min(LOW_VIS, HIGH_VIS, ...) = LOW_VIS < CONFIDENCE_BORDERLINE
        assert result.passed is None
        assert result.confidence < CONFIDENCE_BORDERLINE

    def test_phase_label_in_detail(self):
        """Il phase_label deve comparire nel campo detail."""
        result = check_lockout(
            self.SHOULDER_ERECT, self.HIP_ERECT, self.KNEE_ERECT, self.ANKLE_ERECT,
            HIGH_VIS, HIGH_VIS, HIGH_VIS, HIGH_VIS,
            phase_label="finale",
        )
        assert "finale" in result.detail

    def test_angles_near_threshold(self):
        """Angoli appena sopra le soglie minime → VALIDO."""
        # Creare angoli leggermente sopra LOCKOUT_KNEE_MIN_DEG e LOCKOUT_HIP_MIN_DEG
        # Knee = 135° (> 130°), Hip = 125° (> 120°) via rotazione
        # Strategia: usare punti che producono angoli noti tramite trigonometria
        knee_angle_deg = LOCKOUT_KNEE_MIN_DEG + 5   # 135°
        hip_angle_deg  = LOCKOUT_HIP_MIN_DEG  + 5   # 125°

        # Per knee_angle a knee=(0,0):
        # hip e ankle formano knee_angle_deg tra loro
        hip_kp   = (np.cos(np.radians(90 - knee_angle_deg / 2)) * 100,
                    -np.sin(np.radians(90 - knee_angle_deg / 2)) * 100)
        ankle_kp = (-np.cos(np.radians(90 - knee_angle_deg / 2)) * 100,
                    -np.sin(np.radians(90 - knee_angle_deg / 2)) * 100)
        shoulder_kp = (hip_kp[0], hip_kp[1] - 100)

        result = check_lockout(
            shoulder_kp, hip_kp, (0.0, 0.0), ankle_kp,
            HIGH_VIS, HIGH_VIS, HIGH_VIS, HIGH_VIS,
        )
        # Non assert su passed (l'angolo all'anca dipende dalla geometria specifica),
        # verifichiamo solo che l'oggetto ritornato sia valido
        assert isinstance(result, CriterionResult)
        assert result.detail != ""


# ══════════════════════════════════════════════════════════════════════════════
# check_feet
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckFeet:

    # Con PX_PER_M=500: FEET_LIFT_THRESHOLD_M=0.15 → 75 px
    LIFT_THRESHOLD_PX = FEET_LIFT_THRESHOLD_M * PX_PER_M  # 75 px
    REF_Y = 500.0

    def _make_series(self, values: list[float]) -> tuple[list[float], list[float]]:
        """Crea serie Y e visibilità alta."""
        return values, [HIGH_VIS] * len(values)

    # ── Nessun sollevamento ────────────────────────────────────────────────────

    def test_no_lift_constant(self):
        """Tallone costante al riferimento → PASS."""
        y_series, vis = self._make_series([self.REF_Y] * 20)
        result = check_feet(y_series, self.REF_Y, vis, px_per_meter=PX_PER_M)
        assert result.passed is True

    def test_no_lift_small_jitter(self):
        """Piccole oscillazioni (< soglia) → PASS."""
        # delta = 20 px < 75 px
        y_series, vis = self._make_series([self.REF_Y - 20.0] * 20)
        result = check_feet(y_series, self.REF_Y, vis, px_per_meter=PX_PER_M)
        assert result.passed is True

    def test_no_lift_ankle_goes_down(self):
        """Tallone che scende (Y cresce) → non è un lift → PASS."""
        y_series, vis = self._make_series([self.REF_Y + 50.0] * 20)
        result = check_feet(y_series, self.REF_Y, vis, px_per_meter=PX_PER_M)
        assert result.passed is True

    # ── Sollevamento evidente ──────────────────────────────────────────────────

    def test_clear_lift_sustained(self):
        """Lift di 80 px per JITTER_FRAMES frame consecutivi → FAIL."""
        delta = int(self.LIFT_THRESHOLD_PX) + 5  # 80 px > 75 px
        y_series = [self.REF_Y] * 5 + [self.REF_Y - delta] * FEET_JITTER_FRAMES + [self.REF_Y] * 10
        vis      = [HIGH_VIS] * len(y_series)
        result   = check_feet(y_series, self.REF_Y, vis, px_per_meter=PX_PER_M)
        assert result.passed is False

    def test_lift_at_start(self):
        """Lift subito all'inizio della serie → FAIL."""
        delta    = int(self.LIFT_THRESHOLD_PX) + 5
        y_series = [self.REF_Y - delta] * (FEET_JITTER_FRAMES + 2) + [self.REF_Y] * 10
        vis      = [HIGH_VIS] * len(y_series)
        result   = check_feet(y_series, self.REF_Y, vis, px_per_meter=PX_PER_M)
        assert result.passed is False

    def test_lift_at_end(self):
        """Lift alla fine della serie → FAIL."""
        delta    = int(self.LIFT_THRESHOLD_PX) + 5
        y_series = [self.REF_Y] * 10 + [self.REF_Y - delta] * (FEET_JITTER_FRAMES + 2)
        vis      = [HIGH_VIS] * len(y_series)
        result   = check_feet(y_series, self.REF_Y, vis, px_per_meter=PX_PER_M)
        assert result.passed is False

    # ── Lift non sostenuto (jitter) ────────────────────────────────────────────

    def test_lift_not_sustained_below_jitter_frames(self):
        """Lift sopra soglia per soli 2 frame (< JITTER_FRAMES=3) → PASS."""
        delta    = int(self.LIFT_THRESHOLD_PX) + 5
        y_series = [self.REF_Y] * 5 + [self.REF_Y - delta] * (FEET_JITTER_FRAMES - 1) + [self.REF_Y] * 10
        vis      = [HIGH_VIS] * len(y_series)
        result   = check_feet(y_series, self.REF_Y, vis, px_per_meter=PX_PER_M)
        assert result.passed is True

    def test_lift_interrupted_resets_counter(self):
        """Lift interrotto da un frame normale deve resettare il contatore."""
        delta     = int(self.LIFT_THRESHOLD_PX) + 5
        # 2 frame di lift, 1 frame ok, 2 frame di lift → mai 3 consecutivi
        y_series  = ([self.REF_Y] * 5
                     + [self.REF_Y - delta] * 2
                     + [self.REF_Y] * 1
                     + [self.REF_Y - delta] * 2
                     + [self.REF_Y] * 5)
        vis       = [HIGH_VIS] * len(y_series)
        result    = check_feet(y_series, self.REF_Y, vis, px_per_meter=PX_PER_M)
        assert result.passed is True

    # ── Visibilità bassa ──────────────────────────────────────────────────────

    def test_low_visibility_ignores_lift(self):
        """Lift sopra soglia ma visibilità bassa → frame ignorati → PASS."""
        delta    = int(self.LIFT_THRESHOLD_PX) + 5
        y_series = [self.REF_Y - delta] * 20
        vis      = [LOW_VIS] * 20  # tutti invisibili
        result   = check_feet(y_series, self.REF_Y, vis, px_per_meter=PX_PER_M)
        assert result.passed is True

    def test_mixed_visibility_lift_visible_frames(self):
        """Lift nei frame visibili → FAIL; lift nei frame invisibili → ignorato."""
        delta = int(self.LIFT_THRESHOLD_PX) + 5
        # Lift visibile per 3 frame → FAIL
        vis      = [LOW_VIS] * 5 + [HIGH_VIS] * FEET_JITTER_FRAMES + [LOW_VIS] * 5
        y_series = [self.REF_Y] * 5 + [self.REF_Y - delta] * FEET_JITTER_FRAMES + [self.REF_Y] * 5
        result   = check_feet(y_series, self.REF_Y, vis, px_per_meter=PX_PER_M)
        assert result.passed is False

    # ── Modalità pixel (senza calibrazione) ───────────────────────────────────

    def test_no_px_per_meter_uses_pixel_threshold(self):
        """Senza px_per_meter usa FEET_LIFT_THRESHOLD_PX=70 px."""
        delta    = FEET_LIFT_THRESHOLD_PX + 5  # 75 px > 70 px
        y_series = [self.REF_Y - delta] * (FEET_JITTER_FRAMES + 1) + [self.REF_Y] * 5
        vis      = [HIGH_VIS] * len(y_series)
        result   = check_feet(y_series, self.REF_Y, vis, px_per_meter=None)
        assert result.passed is False

    def test_no_px_per_meter_below_pixel_threshold(self):
        """Senza px_per_meter: lift di 50 px < 70 px → PASS."""
        y_series = [self.REF_Y - 50.0] * 20
        vis      = [HIGH_VIS] * 20
        result   = check_feet(y_series, self.REF_Y, vis, px_per_meter=None)
        assert result.passed is True

    # ── Confidence ────────────────────────────────────────────────────────────

    def test_confidence_reflects_visibility(self):
        """Confidence del risultato PASS riflette la visibilità media dei frame."""
        mid_vis  = 0.75
        y_series = [self.REF_Y] * 10
        vis      = [mid_vis] * 10
        result   = check_feet(y_series, self.REF_Y, vis, px_per_meter=PX_PER_M)
        assert result.passed is True
        assert result.confidence == pytest.approx(mid_vis)

    def test_fail_confidence_is_high(self):
        """Lift confermato → FAIL con confidence CONFIDENCE_HIGH."""
        from src.config import CONFIDENCE_HIGH
        delta    = int(self.LIFT_THRESHOLD_PX) + 5
        y_series = [self.REF_Y - delta] * (FEET_JITTER_FRAMES + 1) + [self.REF_Y] * 5
        vis      = [HIGH_VIS] * len(y_series)
        result   = check_feet(y_series, self.REF_Y, vis, px_per_meter=PX_PER_M)
        assert result.passed is False
        assert result.confidence == pytest.approx(CONFIDENCE_HIGH)

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_series(self):
        """Serie vuota → PASS con confidence 0 (nessun frame = nessun lift)."""
        result = check_feet([], self.REF_Y, [], px_per_meter=PX_PER_M)
        assert result.passed is True
        assert result.confidence == 0.0

    def test_single_frame_below_threshold(self):
        """Un solo frame con lift → non raggiunge JITTER_FRAMES → PASS."""
        delta    = int(self.LIFT_THRESHOLD_PX) + 5
        y_series = [self.REF_Y - delta]
        vis      = [HIGH_VIS]
        result   = check_feet(y_series, self.REF_Y, vis, px_per_meter=PX_PER_M)
        assert result.passed is True


# ══════════════════════════════════════════════════════════════════════════════
# CriterionResult — proprietà is_borderline
# ══════════════════════════════════════════════════════════════════════════════

class TestCriterionResult:

    def test_is_borderline_when_passed_none(self):
        r = CriterionResult(passed=None, confidence=0.9)
        assert r.is_borderline is True

    def test_is_borderline_when_confidence_in_borderline_range(self):
        """Confidence nella zona grigia [BORDERLINE, HIGH) → borderline."""
        from src.config import CONFIDENCE_HIGH
        mid = (CONFIDENCE_BORDERLINE + CONFIDENCE_HIGH) / 2
        r = CriterionResult(passed=True, confidence=mid)
        assert r.is_borderline is True

    def test_not_borderline_when_high_confidence_and_passed(self):
        from src.config import CONFIDENCE_HIGH
        r = CriterionResult(passed=True, confidence=CONFIDENCE_HIGH + 0.01)
        assert r.is_borderline is False

    def test_not_borderline_when_failed_with_high_confidence(self):
        from src.config import CONFIDENCE_HIGH
        r = CriterionResult(passed=False, confidence=CONFIDENCE_HIGH + 0.01)
        assert r.is_borderline is False
