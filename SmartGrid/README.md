# Smart Grid Fault Detection using CNN and RF Signal Processing

## Overview

This project presents a **non-contact Smart Grid Fault Detection System** that classifies electrical power line faults using RF signal processing and Deep Learning.

Instead of connecting sensors directly to high-voltage power lines, the system analyzes the RF emissions generated during faults such as:

- Healthy
- Arcing Fault
- Partial Discharge
- Overload

The RF signals are converted into spectrogram images using FFT, and a Convolutional Neural Network (CNN) is trained to classify the fault type.

---

# Features

- Synthetic IQ Signal Generation
- FFT-based Frequency Spectrum Analysis
- Spectrogram Generation
- CNN-based Fault Classification
- RF Feature Extraction
- Real-time Simulation Pipeline
- Automatic Graph Generation
- Model Evaluation
- Prediction using User Input

---

# Project Workflow

```text
IQ Signal
      │
      ▼
FFT
      │
      ▼
Frequency Spectrum
      │
      ▼
Spectrogram
      │
      ▼
CNN Model
      │
      ▼
Fault Prediction
```

---

# Fault Classes

| Class |
|--------|
| Healthy |
| Arcing |
| Partial Discharge |
| Overload |

---

# Project Structure

```
SmartGridFaultDetection_CNN/

│

├── dataset/

│   ├── iq_signals/

│   ├── fft/

│   ├── spectrograms/

│   ├── processed/

│   └── features/

│

├── graphs/

│

├── models/

│

├── src/

│   ├── config.py

│   ├── generate_iq.py

│   ├── fft_processing.py

│   ├── spectrogram.py

│   ├── feature_extraction.py

│   ├── cnn_model.py

│   ├── train.py

│   ├── evaluate.py

│   ├── predict.py

│   └── realtime.py

│

├── requirements.txt

├── README.md

└── .gitignore
```

---

# Installation

Clone the repository

```bash
git clone <repository_url>
```

Create Virtual Environment

```bash
python3 -m venv venv
```

Activate

Linux

```bash
source venv/bin/activate
```

Windows

```cmd
venv\Scripts\activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Step 1

Generate IQ Signals

```bash
python src/generate_iq.py
```

---

## Step 2

Generate FFT

```bash
python src/fft_processing.py
```

---

## Step 3

Generate Spectrograms

```bash
python src/spectrogram.py
```

---

## Step 4

Train CNN

```bash
python src/train.py
```

---

## Step 5

Evaluate Model

```bash
python src/evaluate.py
```

---

## Step 6

Predict

```bash
python src/predict.py
```

---

## Step 7

Realtime Simulation

```bash
python src/realtime.py
```

---

# Deep Learning Model

The CNN architecture contains

- Convolution Layers
- MaxPooling Layers
- Batch Normalization
- Dropout Layers
- Fully Connected Dense Layers
- Softmax Output Layer

Input Size

```
128 × 128 × 1
```

Output Classes

- Healthy
- Arcing
- Overload
- Partial Discharge

---

# RF Features Extracted

The following RF parameters are extracted from the FFT spectrum.

- Signal Power
- Peak Frequency
- Bandwidth
- Noise Floor
- Spectral Centroid
- Spectral Entropy
- Peak Amplitude
- RMS
- Crest Factor
- Zero Crossings
- Kurtosis
- Skewness
- FFT Peaks
- Signal-to-Noise Ratio (SNR)
- Harmonic Energy

---

# Generated Graphs

The project automatically generates

- Frequency Spectrum
- Spectrogram
- CNN Input Image
- Accuracy Curve
- Loss Curve
- Confusion Matrix
- Prediction Probability Graph

---

# Technologies Used

Programming Language

- Python

Libraries

- TensorFlow
- NumPy
- SciPy
- OpenCV
- Matplotlib
- Pandas
- Scikit-learn

---

# Future Work

The current implementation uses simulated IQ signals.

Future improvements include

- RTL-SDR Integration
- Live RF Signal Acquisition
- Arduino Integration
- GSM Alert System
- LCD Display
- Real-time Fault Monitoring
- Multi-node Fault Localization

---

# Results

The trained CNN model successfully classifies

- Healthy
- Arcing
- Partial Discharge
- Overload

using spectrogram images generated from RF IQ signals.

---

# Author

Developed as an academic project on

**RF-Based Smart Grid Fault Detection using Deep Learning**
