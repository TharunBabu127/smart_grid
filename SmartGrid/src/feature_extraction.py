"""
=========================================================
Feature Extraction from IQ Signals
=========================================================
"""

import os
import numpy as np
import pandas as pd

from scipy.stats import entropy
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.signal import find_peaks

from config import *
from utils import *

create_directories()

# ==========================================================
# Feature Extraction Function
# ==========================================================

def extract_features(signal):

    amplitude = np.abs(signal)

    # -----------------------------
    # FFT
    # -----------------------------

    fft = np.fft.fft(signal)

    fft = np.abs(fft[:len(fft)//2])

    freqs = np.fft.fftfreq(
        len(signal),
        d=1/SAMPLE_RATE
    )[:len(signal)//2]

    # -----------------------------
    # Basic Features
    # -----------------------------

    power = np.mean(amplitude**2)

    rms = np.sqrt(power)

    peak_amplitude = np.max(amplitude)

    mean_amplitude = np.mean(amplitude)

    variance = np.var(amplitude)

    std = np.std(amplitude)

    dynamic_range = peak_amplitude - np.min(amplitude)

    crest_factor = peak_amplitude / rms

    # -----------------------------
    # Zero Crossing
    # -----------------------------

    zero_crossing = np.sum(

        np.diff(np.sign(signal.real)) != 0

    )

    # -----------------------------
    # Frequency Features
    # -----------------------------

    peak_frequency = freqs[np.argmax(fft)]

    spectral_centroid = np.sum(freqs*fft)/np.sum(fft)

    bandwidth = np.sqrt(

        np.sum(

            ((freqs-spectral_centroid)**2)*fft

        ) / np.sum(fft)

    )

    # -----------------------------
    # Spectral Entropy
    # -----------------------------

    probability = fft / np.sum(fft)

    spectral_entropy = entropy(probability)

    # -----------------------------
    # Harmonic Energy
    # -----------------------------

    harmonic_energy = np.sum(

        fft[:100]**2

    )

    # -----------------------------
    # Spectral Flatness
    # -----------------------------

    spectral_flatness = np.exp(

        np.mean(np.log(fft+1e-12))

    ) / np.mean(fft)

    # -----------------------------
    # Roll-off Frequency
    # -----------------------------

    cumulative = np.cumsum(fft)

    rolloff = freqs[

        np.where(

            cumulative >= 0.85*cumulative[-1]

        )[0][0]

    ]

    # -----------------------------
    # Peak Count
    # -----------------------------

    peaks, _ = find_peaks(

        fft,

        height=np.max(fft)*0.30

    )

    peak_count = len(peaks)

    # -----------------------------
    # SNR
    # -----------------------------

    signal_power = np.mean(amplitude**2)

    noise_power = np.var(amplitude)

    snr = 10*np.log10(

        signal_power/(noise_power+1e-12)

    )

    # -----------------------------
    # Statistics
    # -----------------------------

    skewness = skew(amplitude)

    kurt = kurtosis(amplitude)

    return [

        power,

        rms,

        peak_amplitude,

        mean_amplitude,

        variance,

        std,

        dynamic_range,

        crest_factor,

        zero_crossing,

        peak_frequency,

        spectral_centroid,

        bandwidth,

        spectral_entropy,

        harmonic_energy,

        spectral_flatness,

        rolloff,

        peak_count,

        snr,

        skewness,

        kurt

    ]

# ==========================================================
# Column Names
# ==========================================================

columns = [

    "Power",

    "RMS",

    "PeakAmplitude",

    "MeanAmplitude",

    "Variance",

    "Std",

    "DynamicRange",

    "CrestFactor",

    "ZeroCrossing",

    "PeakFrequency",

    "SpectralCentroid",

    "Bandwidth",

    "Entropy",

    "HarmonicEnergy",

    "SpectralFlatness",

    "RollOff",

    "PeakCount",

    "SNR",

    "Skewness",

    "Kurtosis",

    "Label"

]

# ==========================================================
# Process Dataset
# ==========================================================

dataset = []

print("="*60)
print("Extracting RF Features")
print("="*60)

for cls in CLASSES:

    print(f"\nProcessing {cls}")

    folder = os.path.join(IQ_DATASET, cls)

    files = sorted(os.listdir(folder))

    for file in files:

        signal = np.load(

            os.path.join(folder,file)

        )

        features = extract_features(signal)

        features.append(cls)

        dataset.append(features)

# ==========================================================
# Save CSV
# ==========================================================

df = pd.DataFrame(

    dataset,

    columns=columns

)

csv_path = os.path.join(

    FEATURE_DATASET,

    "rf_features.csv"

)

df.to_csv(

    csv_path,

    index=False

)

print("\nFeatures Saved Successfully")

print(csv_path)

print()

print(df.head())

print()

print("Dataset Shape :",df.shape)