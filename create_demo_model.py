"""
Create a demo model for the dashboard that shows varied predictions
This is a workaround until the original model can be properly loaded
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
import numpy as np

print("Creating demo model...")

# Create model with same architecture
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.build((None, 224, 224, 3))

# Initialize with random weights (will give varied predictions)
print("Model created with ImageNet base + random top layers")
print("This will give varied predictions for demo purposes")

# Save
model.save('autism_model_demo.h5')
print("\n✓ Demo model saved as: autism_model_demo.h5")
print("\nNote: This is a demo model with random weights.")
print("For accurate predictions, you need to load your trained model")
print("with a compatible TensorFlow/Keras version (TF 2.15 + Python 3.10)")
