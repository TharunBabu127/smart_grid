"""
====================================================
Hybrid CNN + RF Feature Model
====================================================
"""

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (

    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
    Concatenate

)

from tensorflow.keras.optimizers import Adam

from config import *

# ==========================================================
# Build Hybrid Model
# ==========================================================

def build_hybrid_model():

    # =====================================================
    # Image Input
    # =====================================================

    image_input = Input(

        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1),

        name="Spectrogram_Input"

    )

    x = Conv2D(32,(3,3),activation="relu",padding="same")(image_input)

    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)

    x = Conv2D(64,(3,3),activation="relu",padding="same")(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)

    x = Conv2D(128,(3,3),activation="relu",padding="same")(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D()(x)

    x = Flatten()(x)

    x = Dense(128,activation="relu")(x)

    x = Dropout(0.40)(x)

    # =====================================================
    # RF Feature Input
    # =====================================================

    feature_input = Input(

        shape=(20,),

        name="RF_Features"

    )

    y = Dense(64,activation="relu")(feature_input)

    y = BatchNormalization()(y)

    y = Dropout(0.30)(y)

    y = Dense(32,activation="relu")(y)

    # =====================================================
    # Fusion
    # =====================================================

    merged = Concatenate()([x,y])

    z = Dense(128,activation="relu")(merged)

    z = Dropout(0.40)(z)

    z = Dense(64,activation="relu")(z)

    output = Dense(

        4,

        activation="softmax",

        name="Output"

    )(z)

    model = Model(

        inputs=[image_input,feature_input],

        outputs=output

    )

    model.compile(

        optimizer=Adam(learning_rate=LEARNING_RATE),

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    return model


# ==========================================================
# Test
# ==========================================================

if __name__=="__main__":

    model = build_hybrid_model()

    model.summary()