"""
=========================================================
Evaluate CNN Model
=========================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import *

# ==========================================================
# Load Model
# ==========================================================

print("=" * 60)
print("Loading Trained Model")
print("=" * 60)

model = load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# ==========================================================
# Load Test Dataset
# ==========================================================

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),

    color_mode="grayscale",

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False

)

# ==========================================================
# Evaluate Model
# ==========================================================

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(

    test_generator,

    verbose=1

)

print("\n==============================")

print(f"Test Loss     : {loss:.4f}")

print(f"Test Accuracy : {accuracy*100:.2f}%")

print("==============================")

# ==========================================================
# Predictions
# ==========================================================

predictions = model.predict(test_generator)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes

class_labels = list(test_generator.class_indices.keys())

# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report\n")

print(

    classification_report(

        true_classes,

        predicted_classes,

        target_names=class_labels

    )

)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(

    true_classes,

    predicted_classes

)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=class_labels

)

plt.figure(figsize=(8,8))

disp.plot(

    cmap="Blues",

    values_format="d"

)

plt.title("Confusion Matrix")

plt.savefig(

    os.path.join(

        GRAPH_DIR,

        "confusion_matrix.png"

    ),

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("\nConfusion Matrix Saved Successfully!")

print(

    os.path.join(

        GRAPH_DIR,

        "confusion_matrix.png"

    )

)