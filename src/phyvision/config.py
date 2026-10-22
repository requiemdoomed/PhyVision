"""Central configuration for PhyVision.
"""
from pathlib import Path

PROJECT_NAME = "PhyVision"
VERSION = "5.1.0"
AUTHOR = "Swayam Shakya"

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "phyvision_dataset"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
WEB_DIR = ROOT / "web"
DEMO_DIR = WEB_DIR / "demo"

# Order matters: index i of the model output == CLASSES[i]
CLASSES = ["athletic", "normal", "obese", "skinny"]
CLASS_LABELS = {
    "athletic": "Athletic",
    "normal": "Normal",
    "obese": "Obese",
    "skinny": "Skinny",
}

# Silhouette tensor the network sees (single channel, square)
SIL_SIZE = 128
# Number of horizontal slices in the shape-profile descriptor
PROFILE_BINS = 48
PROFILE_DIM = PROFILE_BINS * 2 + 4

# Dataset image size written by the generator (width, height)
IMG_W, IMG_H = 192, 288

# Default dataset sizes (images per class)
SPLIT_SIZES = {"train": 2000, "val": 1000, "test": 1000}

CHECKPOINT = MODELS_DIR / "phyvision_phynet.pt"
METRICS_JSON = REPORTS_DIR / "metrics.json"
HASH_DB = MODELS_DIR / "train_hashes.json"
