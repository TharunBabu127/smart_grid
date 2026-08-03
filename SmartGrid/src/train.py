"""
=========================================================
Train CNN on Spectrogram Dataset
=========================================================
"""

import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from cnn_model import build_cnn
from config import *

# ==========================================================
# Create Graph Directory
# ==========================================================

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================================
# Image Generators
# ==========================================================

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=10,

    width_shift_range=0.10,

    height_shift_range=0.10,

    zoom_range=0.10,

    horizontal_flip=False,

    fill_mode="nearest"

)

validation_datagen = ImageDataGenerator(

    rescale=1./255

)

# ==========================================================
# Load Dataset
# ==========================================================

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),

    color_mode="grayscale",

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=True

)

import json

with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
    json.dump(train_generator.class_indices, f, indent=4)

print("Class indices saved:")
print(train_generator.class_indices)

validation_generator = validation_datagen.flow_from_directory(

    VALIDATION_DIR,

    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),

    color_mode="grayscale",

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False

)

print("\nClasses Found:")

print(train_generator.class_indices)

# ==========================================================
# Build Model
# ==========================================================

model = build_cnn()

model.summary()

# ==========================================================
# Callbacks
# ==========================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=4,

    verbose=1

)

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

callbacks = [

    early_stop,

    reduce_lr,

    checkpoint

]

# ==========================================================
# Training
# ==========================================================

history = model.fit(

    train_generator,

    validation_data=validation_generator,

    epochs=EPOCHS,

    callbacks=callbacks,

    verbose=1

)

print("\nTraining Completed Successfully!")

# ==========================================================
# Save Final Model
# ==========================================================

model.save(MODEL_PATH)

print("\nModel Saved")

print(MODEL_PATH)

# ==========================================================
# Accuracy Plot
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Train Accuracy")

plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(

    os.path.join(

        GRAPH_DIR,

        "accuracy.png"

    )

)

plt.close()

# ==========================================================
# Loss Plot
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Train Loss")

plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss")

plt.legend()

plt.grid(True)

plt.savefig(

    os.path.join(

        GRAPH_DIR,

        "loss.png"

    )

)

plt.close()

print("\nGraphs Saved Successfully")


