"""
=========================================================
Smart Grid Fault Detection - Real Time Prediction
Workflow

IQ Signal
    ↓
FFT
    ↓
Frequency Spectrum
    ↓
Spectrogram
    ↓
Feature Extraction
    ↓
CNN Prediction
=========================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from scipy.signal import spectrogram
from scipy.stats import kurtosis, skew

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from config import MODEL_PATH, SAMPLE_RATE, GRAPH_DIR, IMAGE_WIDTH, IMAGE_HEIGHT

# =====================================================
# Class Order
# =====================================================

CLASSES = [

    "Arcing",

    "Healthy",

    "Overload",

    "Partial_Discharge"

]

# =====================================================
# Load CNN Model
# =====================================================

print("="*60)
print("Loading CNN Model...")
print("="*60)

model = load_model(MODEL_PATH)

print("Model Loaded Successfully.")

# =====================================================
# User Input
# =====================================================

print("\nExample:")

print("dataset/iq_signals/Healthy/Healthy_0001.npy")

iq_path = input("\nEnter IQ Signal Path : ").strip()

if not os.path.exists(iq_path):

    print("\nIQ File Not Found.")

    sys.exit(1)

# =====================================================
# Load IQ Signal
# =====================================================

print("\nLoading IQ Signal...")

iq_signal = np.load(iq_path)

print("IQ Samples :", len(iq_signal))

# =====================================================
# FFT
# =====================================================

print("\nPerforming FFT...")

fft_result = np.fft.fft(iq_signal)

fft_shift = np.fft.fftshift(fft_result)

magnitude = np.abs(fft_shift)

frequency = np.fft.fftshift(

    np.fft.fftfreq(

        len(iq_signal),

        d=1/SAMPLE_RATE

    )

)

print("FFT Completed.")

# =====================================================
# Frequency Spectrum
# =====================================================

print("\nGenerating Frequency Spectrum...")

plt.figure(figsize=(12,5))

plt.plot(

    frequency/1e6,

    magnitude

)

plt.xlabel("Frequency (MHz)")

plt.ylabel("Magnitude")

plt.title("FFT Frequency Spectrum")

plt.grid(True)

frequency_graph = os.path.join(

    GRAPH_DIR,

    "frequency_spectrum.png"

)

plt.savefig(

    frequency_graph,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("Frequency Spectrum Saved.")

# =====================================================
# Next Part Starts Here
# =====================================================

# =====================================================
# Generate Spectrogram
# =====================================================

print("\nGenerating Spectrogram...")

# Compute spectrogram directly from the IQ signal magnitude
frequencies, times, Sxx = spectrogram(

    np.abs(iq_signal),

    fs=SAMPLE_RATE,

    window="hann",

    nperseg=256,

    noverlap=128,

    nfft=512,

    scaling="density"

)

# Convert to dB scale
Sxx_db = 10 * np.log10(Sxx + 1e-12)

# =====================================================
# Save Spectrogram
# =====================================================

spectrogram_path = os.path.join(

    GRAPH_DIR,

    "current_spectrogram.png"

)

plt.figure(figsize=(6,6))

plt.pcolormesh(

    times,

    frequencies/1e6,

    Sxx_db,

    shading="gouraud"

)

plt.ylabel("Frequency (MHz)")

plt.xlabel("Time (s)")

plt.title("Generated Spectrogram")

plt.colorbar(label="Magnitude (dB)")

plt.tight_layout()

plt.savefig(

    spectrogram_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("Spectrogram Saved Successfully.")

# =====================================================
# Prepare Image for CNN
# =====================================================

print("\nPreparing Image For CNN...")

Sxx_norm = (Sxx_db - Sxx_db.min()) / (Sxx_db.max() - Sxx_db.min())

raw_img = Image.fromarray(np.uint8(Sxx_norm * 255))

raw_img = raw_img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

cnn_input_path = os.path.join(GRAPH_DIR, "cnn_input.png")

raw_img.save(cnn_input_path)

img = image.load_img(

    cnn_input_path,

    color_mode="grayscale",

    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)

)

img_array = image.img_to_array(img)

img_array = img_array.astype("float32")

img_array /= 255.0

img_array = np.expand_dims(

    img_array,

    axis=0

)

print("Image Ready.")

# =====================================================
# Display Generated Spectrogram
# =====================================================

plt.figure(figsize=(6,6))

plt.imshow(

    img,

    cmap="gray"

)

plt.title("CNN Input Spectrogram")

plt.axis("off")

plt.savefig(

    os.path.join(

        GRAPH_DIR,

        "cnn_input.png"

    ),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("CNN Input Image Saved.")

# =====================================================
# Part 3 Starts Below
# =====================================================

# =====================================================
# Feature Extraction
# =====================================================

print("\nExtracting RF Features...")

# -------------------------------
# Power
# -------------------------------

power = np.mean(np.abs(iq_signal) ** 2)

# -------------------------------
# Peak Frequency
# -------------------------------

peak_index = np.argmax(magnitude)

peak_frequency = frequency[peak_index]

# -------------------------------
# Bandwidth (-3 dB)
# -------------------------------

threshold = magnitude.max() / np.sqrt(2)

indices = np.where(magnitude >= threshold)[0]

if len(indices) > 1:

    bandwidth = frequency[indices[-1]] - frequency[indices[0]]

else:

    bandwidth = 0

# -------------------------------
# Noise Floor
# -------------------------------

noise_floor = np.median(magnitude)

# -------------------------------
# Spectral Centroid
# -------------------------------

spectral_centroid = np.sum(
    np.abs(frequency) * magnitude
) / np.sum(magnitude)

# -------------------------------
# Spectral Entropy
# -------------------------------

prob = magnitude / np.sum(magnitude)

entropy = -np.sum(
    prob * np.log2(prob + 1e-12)
)

# -------------------------------
# Peak Amplitude
# -------------------------------

peak_amplitude = np.max(np.abs(iq_signal))

# -------------------------------
# RMS
# -------------------------------

rms = np.sqrt(np.mean(np.abs(iq_signal) ** 2))

# -------------------------------
# Crest Factor
# -------------------------------

crest_factor = peak_amplitude / (rms + 1e-12)

# -------------------------------
# Zero Crossings
# -------------------------------

zero_crossings = np.sum(

    np.diff(

        np.sign(np.real(iq_signal))

    ) != 0

)

# -------------------------------
# Kurtosis
# -------------------------------

kurt = kurtosis(np.real(iq_signal))

# -------------------------------
# Skewness
# -------------------------------

skewness = skew(np.real(iq_signal))

# -------------------------------
# FFT Peaks
# -------------------------------

peak_threshold = magnitude.max() * 0.5

fft_peaks = np.sum(

    magnitude > peak_threshold

)

# -------------------------------
# Signal-to-Noise Ratio
# -------------------------------

signal_power = np.max(magnitude) ** 2

noise_power = np.mean(magnitude ** 2)

snr = 10 * np.log10(

    signal_power /

    (noise_power + 1e-12)

)

# -------------------------------
# Harmonic Energy
# -------------------------------

harmonic_energy = np.sum(

    magnitude ** 2

)

# =====================================================
# Store Features
# =====================================================

features = {

    "Power": power,

    "Peak Frequency (MHz)": peak_frequency / 1e6,

    "Bandwidth (kHz)": bandwidth / 1e3,

    "Noise Floor": noise_floor,

    "Spectral Centroid (MHz)": spectral_centroid / 1e6,

    "Entropy": entropy,

    "Peak Amplitude": peak_amplitude,

    "RMS": rms,

    "Crest Factor": crest_factor,

    "Zero Crossings": zero_crossings,

    "Kurtosis": kurt,

    "Skewness": skewness,

    "FFT Peaks": fft_peaks,

    "SNR (dB)": snr,

    "Harmonic Energy": harmonic_energy

}

# =====================================================
# Print Features
# =====================================================

print("\n" + "="*60)
print("Extracted RF Features")
print("="*60)

for key, value in features.items():

    if isinstance(value, float):

        print(f"{key:25s}: {value:.4f}")

    else:

        print(f"{key:25s}: {value}")

# =====================================================
# Save Features
# =====================================================

feature_df = pd.DataFrame([features])

feature_path = os.path.join(

    GRAPH_DIR,

    "current_features.csv"

)

feature_df.to_csv(

    feature_path,

    index=False

)

print("\nFeatures Saved Successfully.")

print(feature_path)

# =====================================================
# Part 4 Starts Below
# =====================================================
# =====================================================
# CNN Prediction
# =====================================================

print("\n")
print("="*60)
print("Predicting Fault Type...")
print("="*60)

prediction = model.predict(

    img_array,

    verbose=0

)

predicted_index = np.argmax(prediction[0])

confidence = prediction[0][predicted_index]

predicted_class = CLASSES[predicted_index]

# =====================================================
# Print Result
# =====================================================

print("\n")

print("="*60)

print("FAULT DETECTION RESULT")

print("="*60)

print(f"Predicted Fault Type : {predicted_class}")

print(f"Confidence           : {confidence*100:.2f}%")

print("="*60)

print("\nProbability of Every Class\n")

for cls, prob in zip(CLASSES, prediction[0]):

    print(f"{cls:22s}: {prob*100:.2f}%")

# =====================================================
# Prediction Graph
# =====================================================

plt.figure(figsize=(8,5))

plt.bar(

    CLASSES,

    prediction[0]

)

plt.xlabel("Fault Classes")

plt.ylabel("Probability")

plt.title("CNN Prediction")

prediction_graph = os.path.join(

    GRAPH_DIR,

    "prediction.png"

)

plt.savefig(

    prediction_graph,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("\nPrediction Graph Saved.")

# =====================================================
# Final Report
# =====================================================

print("\n")

print("="*70)

print("SMART GRID FAULT DETECTION REPORT")

print("="*70)

print(f"IQ Samples Loaded      : {len(iq_signal)}")

print(f"Sample Rate            : {SAMPLE_RATE/1e6:.2f} MHz")

print(f"FFT Size               : {len(fft_result)}")

print(f"Predicted Fault        : {predicted_class}")

print(f"Prediction Confidence  : {confidence*100:.2f}%")

print()

print("Graphs Generated")

print("-------------------------")

print(f"Frequency Spectrum : {frequency_graph}")

print(f"Spectrogram        : {spectrogram_path}")

print(f"CNN Input Image    : {os.path.join(GRAPH_DIR,'cnn_input.png')}")

print(f"Prediction Graph   : {prediction_graph}")

print(f"Feature CSV        : {feature_path}")

print("="*70)

print("\nRealtime Detection Completed Successfully!")
