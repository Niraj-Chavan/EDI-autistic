"""
Create a demo model that gives varied predictions for demonstration
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
import numpy as np

print("Creating demo model with varied predictions...")

# Create model
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.4),
    Dense(256, activation='relu', name='dense'),
    Dropout(0.4),
    Dense(128, activation='relu', name='dense_1'),
    Dropout(0.3),
    Dense(1, activation='sigmoid', name='dense_2')
])

model.build((None, 224, 224, 3))

# Set custom weights to give more varied predictions
# Adjust the final layer to be more sensitive
for layer in model.layers:
    if layer.name == 'dense_2':
        # Get current weights
        kernel, bias = layer.get_weights()
        # Scale them up to give more varied predictions
        kernel = kernel * 5.0
        bias = np.array([0.0])  # Center around 0.5
        layer.set_weights([kernel, bias])
        print(f"Adjusted {layer.name} weights for varied predictions")

# Test
print("\nTesting predictions:")
from tensorflow.keras.applications.efficientnet import preprocess_input
for i in range(5):
    test_img = np.random.randint(0, 255, (1, 224, 224, 3)).astype(np.float32)
    test_img = preprocess_input(test_img)
    pred = model.predict(test_img, verbose=0)[0][0]
    result = "Autism" if pred > 0.5 else "Non-Autism"
    conf = pred*100 if pred > 0.5 else (1-pred)*100
    print(f"  Test {i+1}: {result} ({conf:.1f}%)")

# Save
model.save('autism_model_demo_varied.h5')
print("\n✓ Demo model saved: autism_model_demo_varied.h5")
print("This model will give more varied predictions for demonstration")
