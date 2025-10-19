"""
Configuration file for facial emotion classification.

Contains all hyperparameters and paths for training, validation, and testing.
"""

import torch
from pathlib import Path

# ============================================================================
# Dataset Configuration
# ============================================================================
DATA_ROOT = "Facial_emotion_images"
TRAIN_DIR = Path(DATA_ROOT) / "train"
VAL_DIR = Path(DATA_ROOT) / "validation"
TEST_DIR = Path(DATA_ROOT) / "test"

# Emotion classes
EMOTION_CLASSES = ["happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTION_CLASSES)

# Class to index mapping
CLASS_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTION_CLASSES)}
IDX_TO_CLASS = {idx: emotion for emotion, idx in CLASS_TO_IDX.items()}

# ============================================================================
# Model Configuration
# ============================================================================
IMAGE_SIZE = 48  # Input image size (48x48) - matches TensorFlow model
INPUT_CHANNELS = 3  # RGB images

# CNN Architecture (simplified model matching TensorFlow specification)
# Conv2D(16) → LeakyReLU → Conv2D(32) → LeakyReLU → MaxPool → Flatten → Dense(32) → Dense(4)
CONV_CHANNELS = [16, 32]  # Number of filters in conv layers
FC_HIDDEN_SIZE = 32  # Hidden layer size in fully connected layer
DROPOUT_RATE = 0.0  # No dropout in the specified architecture

# ============================================================================
# Training Configuration
# ============================================================================
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Optimizer
OPTIMIZER_TYPE = "Adam"  # Options: "Adam", "SGD", "AdamW"
MOMENTUM = 0.9  # For SGD

# Learning rate scheduler
USE_SCHEDULER = True
SCHEDULER_TYPE = "ReduceLROnPlateau"  # Options: "ReduceLROnPlateau", "StepLR"
SCHEDULER_PATIENCE = 5  # For ReduceLROnPlateau
SCHEDULER_FACTOR = 0.5  # Reduce LR by this factor
SCHEDULER_MIN_LR = 1e-6

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

# ============================================================================
# Data Augmentation Configuration
# ============================================================================
# Training augmentation
TRAIN_AUGMENTATION = {
    "horizontal_flip_prob": 0.5,
    "rotation_degrees": 15,
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1
    },
    "random_crop_scale": (0.8, 1.0),
    "random_crop_ratio": (0.9, 1.1)
}

# Normalization (ImageNet statistics)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# ============================================================================
# Hardware Configuration
# ============================================================================
# Automatically select device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # DataLoader workers
PIN_MEMORY = True if torch.cuda.is_available() else False

# ============================================================================
# Model Saving Configuration
# ============================================================================
MODEL_SAVE_DIR = Path("models")
MODEL_SAVE_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = MODEL_SAVE_DIR / "best_model.pth"
LAST_MODEL_PATH = MODEL_SAVE_DIR / "last_model.pth"
HISTORY_PATH = MODEL_SAVE_DIR / "training_history.json"

# ============================================================================
# Results Configuration
# ============================================================================
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
TRAINING_CURVES_PATH = RESULTS_DIR / "training_curves.png"
CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "classification_report.txt"

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_INTERVAL = 10  # Print training stats every N batches
SAVE_INTERVAL = 1  # Save model every N epochs

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Class Weights for Imbalanced Dataset
# ============================================================================
# Based on training set distribution (will be calculated dynamically in train.py)
# happy: 3,976 (26.3%), neutral: 3,978 (26.3%), sad: 3,982 (26.4%), surprise: 3,173 (21.0%)
USE_CLASS_WEIGHTS = True

# ============================================================================
# Helper Functions
# ============================================================================
def print_config():
    """Print all configuration parameters."""
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"\nDataset:")
    print(f"  Data root: {DATA_ROOT}")
    print(f"  Classes: {EMOTION_CLASSES}")
    print(f"  Number of classes: {NUM_CLASSES}")

    print(f"\nModel:")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Architecture: Simplified CNN (TensorFlow-style)")
    print(f"  Conv channels: {CONV_CHANNELS}")
    print(f"  FC hidden size: {FC_HIDDEN_SIZE}")
    print(f"  Dropout rate: {DROPOUT_RATE} (no dropout)")

    print(f"\nTraining:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Number of epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Optimizer: {OPTIMIZER_TYPE}")
    print(f"  Use scheduler: {USE_SCHEDULER}")
    print(f"  Use early stopping: {USE_EARLY_STOPPING}")
    print(f"  Use class weights: {USE_CLASS_WEIGHTS}")

    print(f"\nHardware:")
    print(f"  Device: {DEVICE}")
    print(f"  Number of workers: {NUM_WORKERS}")
    print(f"  Pin memory: {PIN_MEMORY}")

    print(f"\nPaths:")
    print(f"  Best model: {BEST_MODEL_PATH}")
    print(f"  Results dir: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
