"""
=========================================================
Dataset Preprocessing
=========================================================
"""

import os
import random
import shutil

from tqdm import tqdm

from config import *
from utils import *

create_directories()

random.seed(RANDOM_STATE)

print("="*60)
print("Dataset Preprocessing")
print("="*60)

# --------------------------------------------------------

for cls in CLASSES:

    print(f"\nProcessing {cls}")

    source = os.path.join(

        SPECTROGRAM_DATASET,

        cls

    )

    images = sorted(os.listdir(source))

    random.shuffle(images)

    total = len(images)

    train_size = int(0.70*total)

    validation_size = int(0.15*total)

    train = images[:train_size]

    validation = images[
        train_size:
        train_size+validation_size
    ]

    test = images[
        train_size+validation_size:
    ]

    # ---------------------------------------

    for img in tqdm(train):

        shutil.copy(

            os.path.join(source,img),

            os.path.join(TRAIN_DIR,cls,img)

        )

    # ---------------------------------------

    for img in tqdm(validation):

        shutil.copy(

            os.path.join(source,img),

            os.path.join(VALIDATION_DIR,cls,img)

        )

    # ---------------------------------------

    for img in tqdm(test):

        shutil.copy(

            os.path.join(source,img),

            os.path.join(TEST_DIR,cls,img)

        )

print("\nDataset Split Completed Successfully.")