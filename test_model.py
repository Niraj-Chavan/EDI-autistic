import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load the converted model
model = tf.keras.models.load_model('autism_model_keras3.h5', compile=False)

print("Model loaded successfully!")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# Create a random test image
test_img = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
test_img_preprocessed = preprocess_input(test_img.astype(np.float32))

# Make prediction
prediction = model.predict(test_img_preprocessed, verbose=0)
print(f"\nTest prediction: {prediction[0][0]}")

# Test with different random images
print("\nTesting with 5 random images:")
for i in range(5):
    test_img = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
    test_img_preprocessed = preprocess_input(test_img.astype(np.float32))
    pred = model.predict(test_img_preprocessed, verbose=0)[0][0]
    print(f"  Image {i+1}: {pred:.6f}")

# Check if model weights are actually different
print("\nChecking Dense layer weights:")
for layer in model.layers:
    if 'dense' in layer.name:
        weights = layer.get_weights()
        if weights:
            print(f"  {layer.name}: kernel mean = {np.mean(weights[0]):.6f}, bias mean = {np.mean(weights[1]):.6f}")
