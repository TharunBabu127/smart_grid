from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *

generator = ImageDataGenerator(rescale=1./255)

data = generator.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    color_mode="grayscale",
    class_mode="categorical"
)

print(data.class_indices)