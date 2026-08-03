"""
=========================================================
Predict Single Spectrogram Image
=========================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from config import *

# =====================================================
# Class Order (Must Match Training)
# =====================================================

CLASSES = [
    "Arcing",
    "Healthy",
    "Overload",
    "Partial_Discharge"
]

# =====================================================
# Load Model
# =====================================================

print("=" * 60)
print("Loading CNN Model...")
print("=" * 60)

model = load_model(MODEL_PATH)

print("Model Loaded Successfully!")

# =====================================================
# Input Image
# =====================================================

image_path = input("\nEnter Spectrogram Image Path : ").strip()

if not os.path.exists(image_path):

    print("\nImage Not Found!")

    exit()

# =====================================================
# Load Image
# =====================================================

img = image.load_img(

    image_path,

    color_mode="grayscale",

    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH)

)

img_array = image.img_to_array(img)

img_array = img_array.astype("float32") / 255.0

img_array = np.expand_dims(img_array, axis=0)

# =====================================================
# Predict
# =====================================================

prediction = model.predict(img_array, verbose=0)

predicted_index = np.argmax(prediction[0])

confidence = prediction[0][predicted_index]

predicted_class = CLASSES[predicted_index]

# =====================================================
# Print Result
# =====================================================

print("\n" + "=" * 50)

print("Prediction Result")

print("=" * 50)

print(f"Predicted Fault : {predicted_class}")

print(f"Confidence      : {confidence*100:.2f}%")

print("=" * 50)

print("\nClass Probabilities\n")

for cls, prob in zip(CLASSES, prediction[0]):

    print(f"{cls:20s} : {prob*100:.2f}%")

# =====================================================
# Display Image
# =====================================================

plt.figure(figsize=(6,6))

plt.imshow(img, cmap="gray")

plt.title(
    f"{predicted_class}\nConfidence : {confidence*100:.2f}%"
)

plt.axis("off")

plt.savefig(
    os.path.join(GRAPH_DIR, "prediction.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Prediction graph saved successfully!")