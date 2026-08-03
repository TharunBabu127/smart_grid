"""
=========================================================
FFT Processing
=========================================================
"""

import os

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import windows

from tqdm import tqdm

from config import *

from utils import *

create_directories()

# ==========================================================
# FFT Function
# ==========================================================

def compute_fft(signal):

    # Apply Hann Window
    window = windows.hann(len(signal))

    windowed_signal = signal * window

    # FFT
    fft = np.fft.fft(windowed_signal, FFT_SIZE)

    # Shift DC to Center
    fft = np.fft.fftshift(fft)

    # Magnitude
    magnitude = np.abs(fft)

    # Normalize
    magnitude = magnitude / np.max(magnitude)

    # Frequency Axis
    frequency = np.fft.fftshift(
        np.fft.fftfreq(
            FFT_SIZE,
            d=1/SAMPLE_RATE
        )
    )

    return frequency, magnitude

# ==========================================================
# Process All Signals
# ==========================================================

print("="*60)
print("FFT PROCESSING STARTED")
print("="*60)

for cls in CLASSES:

    print(f"\nProcessing {cls}")

    folder = os.path.join(IQ_DATASET, cls)

    files = sorted(os.listdir(folder))

    for file in tqdm(files):

        path = os.path.join(folder, file)

        signal = np.load(path)

        frequency, magnitude = compute_fft(signal)

        filename = file.replace(".npy", "")

        # Save FFT Array
        np.save(

            os.path.join(

                FFT_NUMPY_DIR,

                cls,

                filename + ".npy"

            ),

            magnitude

        )

        # Save Plot
        plt.figure(figsize=(8,4))

        plt.plot(frequency/1e6, magnitude)

        plt.title(filename)

        plt.xlabel("Frequency (MHz)")

        plt.ylabel("Magnitude")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                FFT_IMAGE_DIR,

                cls,

                filename + ".png"

            ),

            dpi=120

        )

        plt.close()

print("\nFFT Processing Completed.")