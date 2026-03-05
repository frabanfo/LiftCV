# ─────────────────────────────────────────────
# Confidence thresholds
# ─────────────────────────────────────────────
CONFIDENCE_HIGH       = 0.85   # decisione emessa (valida / non valida)
CONFIDENCE_BORDERLINE = 0.60   # "borderline — non determinabile con certezza"
# sotto CONFIDENCE_BORDERLINE → "analisi non affidabile"

# ─────────────────────────────────────────────
# IPF criteria parameters
# ─────────────────────────────────────────────
DEPTH_THRESHOLD_DEG   = -6.0   # Compensazione offset anatomico: MediaPipe usa il
                                # centro del giunto (anca e ginocchio), ma il criterio
                                # IPF confronta la piega dell'anca con la sommità della
                                # rotula (~3-5 cm sopra il centro del ginocchio).
                                # A parallela IPF reale, l'angolo calcolato è circa
                                # -4° a -6°. Soglia -6° = valido quando alla parallela
                                # IPF o più in profondità.
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
# Lockout thresholds (angoli geometrici)
# ─────────────────────────────────────────────
LOCKOUT_KNEE_MIN_DEG = 165.0   # angolo hip→knee→ankle per gamba tesa
LOCKOUT_HIP_MIN_DEG  = 160.0   # angolo shoulder→hip→knee per anca estesa

# ─────────────────────────────────────────────
# Scala antropometrica (Drillis & Contini 1966)
# ─────────────────────────────────────────────
TIBIA_HEIGHT_RATIO = 0.246     # tibia / altezza totale

# ─────────────────────────────────────────────
# VBT — curva Gonzalez-Badillo & Sanchez-Medina (2010), back squat
# %1RM = 100 * (1 - A * v^B)
#
# Parametri ricavati da regressione sui dati pubblicati (MPV → %1RM):
#   ~90% @ 0.41 m/s | ~60% @ 0.96 m/s | ~30% @ 1.48 m/s
# Errore atteso: ±8–12% su curva generica non calibrata
# ─────────────────────────────────────────────
VBT_A = 0.409
VBT_B = 1.500
VBT_ERROR_PCT = 10  # dichiarato nell'output
