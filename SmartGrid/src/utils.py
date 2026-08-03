"""
===========================================================
Utility Functions
===========================================================
"""

import os

import numpy as np

import matplotlib.pyplot as plt

from config import *

# ==========================================================
# Create Project Directories
# ==========================================================

def create_directories():

    folders = [

        DATASET_DIR,

        IQ_DATASET,

        SPECTROGRAM_DATASET,

        FEATURE_DATASET,

        PROCESSED_DATASET,

        MODEL_DIR,

        GRAPH_DIR,
        FFT_DATASET,

        FFT_IMAGE_DIR,

        FFT_NUMPY_DIR,
        TRAIN_DIR,

        VALIDATION_DIR,

        TEST_DIR,

    ]

    # ------------------------------------------------------

    for folder in folders:

        os.makedirs(folder, exist_ok=True)

    # ------------------------------------------------------
    # Create Class Folders
    # ------------------------------------------------------

    for cls in CLASSES:

        os.makedirs(os.path.join(IQ_DATASET, cls), exist_ok=True)

        os.makedirs(os.path.join(SPECTROGRAM_DATASET, cls), exist_ok=True)
        os.makedirs(os.path.join(FFT_IMAGE_DIR, cls), exist_ok=True)

        os.makedirs(os.path.join(FFT_NUMPY_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)

        os.makedirs(os.path.join(VALIDATION_DIR, cls), exist_ok=True)

        os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

    print("All Directories Created Successfully.")

# ==========================================================
# Save IQ Signal
# ==========================================================

def save_iq_signal(signal, class_name, filename):

    filepath = os.path.join(

        IQ_DATASET,

        class_name,

        filename + IQ_EXTENSION

    )

    np.save(filepath, signal)

# ==========================================================
# Load IQ Signal
# ==========================================================

def load_iq_signal(filepath):

    return np.load(filepath)

# ==========================================================
# Save Spectrogram
# ==========================================================

def save_spectrogram(image, class_name, filename):

    filepath = os.path.join(

        SPECTROGRAM_DATASET,

        class_name,

        filename + IMAGE_FORMAT

    )

    plt.imsave(filepath, image, cmap="viridis")

# ==========================================================
# Plot Signal
# ==========================================================

def plot_signal(signal, title="IQ Signal"):

    plt.figure(figsize=(10,4))

    plt.plot(signal.real, label="I")

    plt.plot(signal.imag, label="Q")

    plt.title(title)

    plt.xlabel("Samples")

    plt.ylabel("Amplitude")

    plt.grid(True)

    plt.legend()

    plt.show()

# ==========================================================
# Plot Spectrum
# ==========================================================

def plot_fft(freq, magnitude, title="FFT Spectrum"):

    plt.figure(figsize=(10,4))

    plt.plot(freq, magnitude)

    plt.title(title)

    plt.xlabel("Frequency (Hz)")

    plt.ylabel("Magnitude")

    plt.grid(True)

    plt.show()

# ==========================================================
# Normalize Image
# ==========================================================

def normalize_image(image):

    image = image.astype(np.float32)

    image = image - image.min()

    image = image / (image.max() + 1e-10)

    return image

# ==========================================================
# Print Project Information
# ==========================================================

def project_summary():

    print("="*60)

    print("Smart Grid RF Fault Detection")

    print("="*60)

    print(f"Classes           : {CLASSES}")

    print(f"Signals/Class     : {SAMPLES_PER_CLASS}")

    print(f"Total Signals     : {TOTAL_SIGNALS}")

    print(f"Signal Length     : {SIGNAL_LENGTH}")

    print(f"Sample Rate       : {SAMPLE_RATE}")

    print(f"FFT Size          : {FFT_SIZE}")

    print(f"Image Size        : {IMAGE_HEIGHT} x {IMAGE_WIDTH}")

    print("="*60)