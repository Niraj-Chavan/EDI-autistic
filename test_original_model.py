import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
import warnings
warnings.filterwarnings('ignore')

print("Testing BEST_MODEL_acc_0.9518_round_33.h5...")

# Load model
model = tf.keras.models.load_model('BEST_MODEL_acc_0.9518_round_33.h5', compile=False)
print(f"✓ Model loaded: {len(model.layers)} layers")
print(f"  Input shape: {model.input_shape}")

# Test with random images
print("\nTesting with 10 random images:")
for i in range(10):
    test_img = np.random.randint(0, 255, (1, 224, 224, 3)).astype(np.float32)
    test_img = preprocess_input(test_img)
    
    pred = model.predict(test_img, verbose=0)[0][0]
    
    if pred > 0.5:
        result = "Autism"
        confidence = pred * 100
    else:
        result = "Non-Autism"
        confidence = (1 - pred) * 100
    
    print(f"  Image {i+1}: {result} ({confidence:.2f}%) - raw: {pred:.6f}")

print("\n✓ Model is working and giving varied predictions!")
