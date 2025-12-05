import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
import warnings
warnings.filterwarnings('ignore')

print("Testing model predictions...")

# Load the working model
model = tf.keras.models.load_model('autism_model_working.h5', compile=False)
print(f"Model loaded: {len(model.layers)} layers")

# Test with multiple random images
print("\nTesting with 10 random images:")
for i in range(10):
    # Create random image
    test_img = np.random.randint(0, 255, (1, 224, 224, 3)).astype(np.float32)
    test_img_preprocessed = preprocess_input(test_img)
    
    # Predict
    pred = model.predict(test_img_preprocessed, verbose=0)[0][0]
    
    if pred > 0.5:
        result = "Autism"
        confidence = pred * 100
    else:
        result = "Non-Autism"
        confidence = (1 - pred) * 100
    
    print(f"Image {i+1}: {result} ({confidence:.2f}% confidence) - raw: {pred:.6f}")

# Check model weights
print("\nChecking Dense layer weights:")
for layer in model.layers:
    if 'dense' in layer.name:
        weights = layer.get_weights()
        if weights:
            print(f"  {layer.name}: kernel mean = {np.mean(weights[0]):.6f}, bias mean = {np.mean(weights[1]):.6f}")
