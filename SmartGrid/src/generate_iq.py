"""
=========================================================
Generate Synthetic IQ Signals
=========================================================
"""

import numpy as np
from tqdm import tqdm

from config import *
from utils import *

# ==========================================================
# Initialization
# ==========================================================

np.random.seed(RANDOM_STATE)

create_directories()

# ==========================================================
# Time Axis
# ==========================================================

t = np.arange(SIGNAL_LENGTH) / SAMPLE_RATE

# ==========================================================
# Healthy Signal
# ==========================================================

def generate_healthy():

    carrier_freq = np.random.uniform(100e3, 250e3)

    amplitude = np.random.uniform(0.8,1.0)

    phase = np.random.uniform(0,2*np.pi)

    signal = amplitude * np.exp(
        1j*(2*np.pi*carrier_freq*t + phase)
    )

    noise = NOISE_STD * (
        np.random.randn(SIGNAL_LENGTH) +
        1j*np.random.randn(SIGNAL_LENGTH)
    )

    return signal + noise

# ==========================================================
# Partial Discharge
# ==========================================================

def generate_partial_discharge():

    signal = generate_healthy()

    impulse_count = np.random.randint(15,35)

    for _ in range(impulse_count):

        idx = np.random.randint(0,SIGNAL_LENGTH)

        signal[idx] += np.random.uniform(1.5,3.0) * np.exp(
            1j*np.random.uniform(0,2*np.pi)
        )

    return signal

# ==========================================================
# Arcing Fault
# ==========================================================

def generate_arcing():

    signal = generate_healthy()

    burst_count = np.random.randint(8,15)

    for _ in range(burst_count):

        start = np.random.randint(0,SIGNAL_LENGTH-100)

        length = np.random.randint(30,100)

        burst = (

            np.random.randn(length) +

            1j*np.random.randn(length)

        ) * np.random.uniform(1.5,2.5)

        signal[start:start+length] += burst

    return signal

# ==========================================================
# Overload
# ==========================================================

def generate_overload():

    carrier_freq = np.random.uniform(100e3,250e3)

    amplitude = np.random.uniform(1.2,1.8)

    phase = np.random.uniform(0,2*np.pi)

    signal = amplitude*np.exp(

        1j*(2*np.pi*carrier_freq*t + phase)

    )

    # Harmonics

    harmonic1 = 0.40*np.exp(

        1j*(2*np.pi*2*carrier_freq*t)

    )

    harmonic2 = 0.20*np.exp(

        1j*(2*np.pi*3*carrier_freq*t)

    )

    noise = 0.12*(

        np.random.randn(SIGNAL_LENGTH)

        +

        1j*np.random.randn(SIGNAL_LENGTH)

    )

    return signal + harmonic1 + harmonic2 + noise

# ==========================================================
# Save Dataset
# ==========================================================

print("="*60)

print("Generating IQ Dataset")

print("="*60)

for cls in CLASSES:

    print(f"\nGenerating {cls} Signals...")

    for i in tqdm(range(SAMPLES_PER_CLASS)):

        if cls=="Healthy":

            signal = generate_healthy()

        elif cls=="Partial_Discharge":

            signal = generate_partial_discharge()

        elif cls=="Arcing":

            signal = generate_arcing()

        else:

            signal = generate_overload()

        filename = f"{cls}_{i:04d}"

        save_iq_signal(

            signal,

            cls,

            filename

        )

print("\nDataset Generation Completed Successfully!")

print(f"\nTotal Signals : {TOTAL_SIGNALS}")