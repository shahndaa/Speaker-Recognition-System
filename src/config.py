"""
Central configuration for the Speaker Recognition System.
Keeping all tunable values here avoids hardcoded magic numbers/paths
scattered across the codebase.
"""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "16000_pcm_speeches"
NOISE_DIR_NAME = "_background_noise_"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

MODEL_PATH = MODELS_DIR / "speaker_recognition_model.keras"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.json"
TRAINING_HISTORY_PATH = MODELS_DIR / "training_history.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = ASSETS_DIR / "confusion_matrix.png"
TRAINING_CURVES_PATH = ASSETS_DIR / "training_curves.png"

# --- Audio settings ---
SAMPLE_RATE = 16000          # Hz, fixed by the dataset (1 second clips)
SAMPLES_PER_TRACK = 16000    # 1 second at 16kHz

# --- Noise augmentation ---
NOISE_SCALE_MIN = 0.1
NOISE_SCALE_MAX = 0.4
NOISE_AUGMENT_PROBABILITY = 0.6  # chance a training sample gets noise mixed in

# --- Train/val/test split ---
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# --- Model / training ---
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 6
# Config 
# Fix 
# Config 
