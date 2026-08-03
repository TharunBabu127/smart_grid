"""
=========================================================
CNN Model Architecture
=========================================================
"""

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Dropout,
    GlobalAveragePooling2D,
    Dense
)

from tensorflow.keras.optimizers import Adam

from config import *

# ==========================================================
# Build CNN
# ==========================================================

def build_cnn():

    inputs = Input(
        shape=(IMAGE_HEIGHT,
               IMAGE_WIDTH,
               1)
    )

    # ------------------------------------------------------
    # Block 1
    # ------------------------------------------------------

    x = Conv2D(
        32,
        (3,3),
        activation="relu",
        padding="same"
    )(inputs)

    x = BatchNormalization()(x)

    x = MaxPooling2D((2,2))(x)

    x = Dropout(0.25)(x)

    # ------------------------------------------------------
    # Block 2
    # ------------------------------------------------------

    x = Conv2D(
        64,
        (3,3),
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D((2,2))(x)

    x = Dropout(0.30)(x)

    # ------------------------------------------------------
    # Block 3
    # ------------------------------------------------------

    x = Conv2D(
        128,
        (3,3),
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D((2,2))(x)

    x = Dropout(0.35)(x)

    # ------------------------------------------------------
    # Block 4
    # ------------------------------------------------------

    x = Conv2D(
        256,
        (3,3),
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)

    # ------------------------------------------------------
    # Dense Layers
    # ------------------------------------------------------

    x = Dense(
        256,
        activation="relu"
    )(x)

    x = Dropout(0.50)(x)

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = Dropout(0.30)(x)

    outputs = Dense(
        TOTAL_CLASSES,
        activation="softmax"
    )(x)

    # ------------------------------------------------------

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(

        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


# ==========================================================
# Run Standalone
# ==========================================================

if __name__ == "__main__":

    model = build_cnn()

    model.summary()