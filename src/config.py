# ─────────────────────────────────────────────
# Confidence thresholds
# ─────────────────────────────────────────────
CONFIDENCE_HIGH       = 0.80   # decisione emessa ad alta confidenza (valida / non valida).
                                # MediaPipe visibility >= 0.80 è una rilevazione affidabile.
CONFIDENCE_BORDERLINE = 0.60   # "borderline — non determinabile con certezza"
# sotto CONFIDENCE_BORDERLINE → "analisi non affidabile"

# ─────────────────────────────────────────────
# IPF criteria parameters
# ─────────────────────────────────────────────
DEPTH_OFFSET_M        = 0.15   # Distanza verticale massima ammessa tra centro anca
                                # (acetabolo, landmark MediaPipe) e centro ginocchio
                                # perché lo squat sia considerato alla parallela IPF.
                                # Anatomia: acetabolo → piega anca ≈ 10-12 cm;
                                #           sommità rotula → centro ginocchio ≈ 3-5 cm;
                                #           totale atteso a parallela IPF: ~13-17 cm.
                                # 0.15 m è il valore calibrato empiricamente su riprese
                                # laterali standard (camera ~1 m, altezza atleta).
DEPTH_TOLERANCE_M     = 0.02   # ±2 cm → zona borderline attorno alla soglia

# Legacy (non più usato dalla pipeline calibrata, mantenuto per fallback senza px/m)
DEPTH_THRESHOLD_DEG   = -8.0
DEPTH_TOLERANCE_DEG   = 2.0

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
LOCKOUT_KNEE_MIN_DEG = 130.0   # angolo hip→knee→ankle per gamba tesa.
                                # Nota: in ripresa laterale l'anca è arretrata rispetto
                                # alla caviglia → angolo geometrico a knee ~140-155° anche
                                # a gamba tesa. Soglia 130° distingue lockout (>130°) da
                                # squat bottom (~80-90°).
LOCKOUT_HIP_MIN_DEG  = 120.0   # angolo shoulder→hip→knee per anca estesa.
                                # In ripresa laterale ~125-135° a piena estensione.

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
