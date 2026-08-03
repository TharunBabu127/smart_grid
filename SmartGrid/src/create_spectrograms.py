"""
===========================================================
Create Spectrogram Images from IQ Signals
===========================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import spectrogram
from PIL import Image
from tqdm import tqdm

from config import *
from utils import *

create_directories()

print("=" * 60)
print("Generating Spectrogram Images")
print("=" * 60)

# ----------------------------------------------------------

for cls in CLASSES:

    print(f"\nProcessing {cls}")

    input_folder = os.path.join(IQ_DATASET, cls)

    output_folder = os.path.join(SPECTROGRAM_DATASET, cls)

    files = sorted(os.listdir(input_folder))

    for file in tqdm(files):

        filepath = os.path.join(input_folder, file)

        signal = np.load(filepath)

        # -----------------------------------------------
        # Magnitude of IQ Signal
        # -----------------------------------------------

        amplitude = np.abs(signal)

        # -----------------------------------------------
        # STFT Spectrogram
        # -----------------------------------------------

        frequencies, times, Sxx = spectrogram(

            amplitude,

            fs=SAMPLE_RATE,

            window='hann',

            nperseg=256,

            noverlap=128,

            nfft=512,

            scaling='density'

        )

        # -----------------------------------------------
        # Convert to dB
        # -----------------------------------------------

        Sxx = 10 * np.log10(Sxx + 1e-12)

        # -----------------------------------------------
        # Normalize
        # -----------------------------------------------

        Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())

        # -----------------------------------------------
        # Convert to Image
        # -----------------------------------------------

        image = Image.fromarray(

            np.uint8(Sxx * 255)

        )

        image = image.resize(

            (IMAGE_WIDTH, IMAGE_HEIGHT)

        )

        filename = file.replace(".npy", ".png")

        image.save(

            os.path.join(

                output_folder,

                filename

            )

        )

print("\nSpectrogram Generation Completed Successfully.")