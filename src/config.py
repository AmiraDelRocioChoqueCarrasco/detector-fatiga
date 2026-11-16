"""Configuración del proyecto en un solo lugar.

Las rutas, la semilla y los hiperparámetros los dejo aquí para no andar
buscando valores regados por todos los scripts.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MRL_RAW = DATA_DIR / "raw" / "mrl_eye"            # donde descomprimo MRL Eye
VIDEOS_RAW = DATA_DIR / "raw" / "videos"          # videos de YawDD / UTA-RLDD / NTHU
PROCESSED = DATA_DIR / "processed"
MRL_MANIFEST = PROCESSED / "mrl_manifest.csv"     # imagen -> etiqueta -> sujeto -> split
VIDEO_FEATURES = PROCESSED / "video_features.csv" # EAR/MAR por fotograma
MODELS_DIR = ROOT / "models"

SEED = 42
SPLIT = (0.70, 0.15, 0.15)       # train / val / test, siempre separado por sujeto

# --- CNN de ojo ---
IMG_SIZE = 96
BATCH_SIZE = 64
EPOCHS_HEAD = 8                  # fase 1: entreno solo la cabeza
EPOCHS_FINETUNE = 10             # fase 2: ajuste fino
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5
UNFREEZE_LAST_N = 30             # capas del backbone que descongelo

# --- Modelo temporal ---
FPS = 30
WINDOW_SECONDS = 3
STRIDE_SECONDS = 1
FEATURES = ["ear_norm", "mar", "ear_delta"]

# --- Demo en tiempo real (valores iniciales, hay que ajustarlos con datos reales) ---
CALIBRATION_SECONDS = 10
EAR_THRESHOLD_FACTOR = 0.75      # umbral = factor * EAR mediano con ojos abiertos
MICROSLEEP_SECONDS = 1.0         # ojos cerrados de forma continua
MAR_YAWN_THRESHOLD = 0.60
YAWN_SECONDS = 1.5
PERCLOS_WINDOW_SECONDS = 60
PERCLOS_ALERT = 0.15
