"""
===========================================================
Project Configuration File
===========================================================
"""

import os

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==========================================================
# Dataset Folders
# ==========================================================

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

IQ_DATASET = os.path.join(DATASET_DIR, "iq_signals")

SPECTROGRAM_DATASET = os.path.join(DATASET_DIR, "spectrograms")

FEATURE_DATASET = os.path.join(DATASET_DIR, "features")

PROCESSED_DATASET = os.path.join(DATASET_DIR, "processed")
# Dataset folders
TRAIN_DIR = os.path.join(PROCESSED_DATASET, "train")
VALIDATION_DIR = os.path.join(PROCESSED_DATASET, "validation")
TEST_DIR = os.path.join(PROCESSED_DATASET, "test")

# ==========================================================
# Model Folder
# ==========================================================

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.keras")

# ==========================================================
# Graph Folder
# ==========================================================

GRAPH_DIR = os.path.join(PROJECT_ROOT, "graphs")

# ==========================================================
# FFT Folder
# ==========================================================

FFT_DATASET = os.path.join(DATASET_DIR, "fft")

FFT_IMAGE_DIR = os.path.join(FFT_DATASET, "images")

FFT_NUMPY_DIR = os.path.join(FFT_DATASET, "arrays")
# ==========================================================
# Classes
# ==========================================================

CLASSES = [

    "Healthy",

    "Arcing",

    "Partial_Discharge",

    "Overload"

]

# ==========================================================
# Dataset Parameters
# ==========================================================

SAMPLES_PER_CLASS = 1000

TOTAL_CLASSES = len(CLASSES)

TOTAL_SIGNALS = SAMPLES_PER_CLASS * TOTAL_CLASSES

# ==========================================================
# IQ Signal Parameters
# ==========================================================

SAMPLE_RATE = 2_000_000          # 2 MHz

SIGNAL_LENGTH = 4096             # IQ samples

CENTER_FREQUENCY = 150e6         # 150 MHz

NOISE_STD = 0.08

# ==========================================================
# FFT Parameters
# ==========================================================

FFT_SIZE = 1024

WINDOW = "hann"

# ==========================================================
# Spectrogram Parameters
# ==========================================================

IMAGE_HEIGHT = 128

IMAGE_WIDTH = 128

DPI = 100

# ==========================================================
# Training Parameters
# ==========================================================

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 0.0005

VALIDATION_SPLIT = 0.15

TEST_SPLIT = 0.15

RANDOM_STATE = 42

# ==========================================================
# Image Extension
# ==========================================================

IMAGE_FORMAT = ".png"

# ==========================================================
# IQ Extension
# ==========================================================

IQ_EXTENSION = ".npy"

print("Configuration Loaded Successfully.")