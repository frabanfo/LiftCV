# ─────────────────────────────────────────────
# Confidence thresholds
# ─────────────────────────────────────────────
CONFIDENCE_HIGH       = 0.85   # decisione emessa (valida / non valida)
CONFIDENCE_BORDERLINE = 0.60   # "borderline — non determinabile con certezza"
# sotto CONFIDENCE_BORDERLINE → "analisi non affidabile"

# ─────────────────────────────────────────────
# IPF criteria parameters
# ─────────────────────────────────────────────
DEPTH_THRESHOLD_DEG   = 0.0    # 0° = piano coscia == piano ginocchio (parallela)
DEPTH_TOLERANCE_DEG   = 2.0    # ±2° → zona borderline, non KO automatico

FEET_JITTER_FRAMES    = 3      # frame consecutivi minimi per flag sollevamento piedi

# ─────────────────────────────────────────────
# Video input requirements
# ─────────────────────────────────────────────
MIN_FPS               = 30

# ─────────────────────────────────────────────
# Bar tracking
# ─────────────────────────────────────────────
BAR_OCCLUSION_MAX_FRAMES = 10  # oltre questa soglia → metriche barra = N/D

# ─────────────────────────────────────────────
# VBT — curva Gonzalez-Badillo (squat)
# %1RM = 100 * (1 - a * velocity^b)  — parametri pubblicati
# Errore atteso: ±8–12% su curva generica non calibrata
# ─────────────────────────────────────────────
VBT_A = 0.9886
VBT_B = 0.2156
VBT_ERROR_PCT = 10  # dichiarato nell'output
