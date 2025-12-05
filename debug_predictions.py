import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
import warnings
warnings.filterwarnings('ignore')

print("Debugging 87.00.h5 model predictions...")

# Load model
model = tf.keras.models.load_model('87.00.h5', compile=False)
print(f"Model loaded: {model.input_shape} -> {model.output_shape}")

# Test with 10 different random images
print("\nTesting with 10 random images:")
for i in range(10):
    # Create random image
    test_img = np.random.randint(0, 255, (1, 224, 224, 3)).astype(np.float32)
    test_img_preprocessed = preprocess_input(test_img)
    
    # Predict
    pred = model.predict(test_img_preprocessed, verbose=0)
    
    if len(pred[0]) == 2:
        non_autism = pred[0][0]
        autism = pred[0][1]
        result = "Autism" if autism > non_autism else "Non-Autism"
        conf = max(autism, non_autism) * 100
        print(f"  {i+1}. {result} ({conf:.1f}%) - Autism: {autism:.4f}, Non-Autism: {non_autism:.4f}")
    else:
        prob = pred[0][0]
        result = "Autism" if prob > 0.5 else "Non-Autism"
        print(f"  {i+1}. {result} - prob: {prob:.4f}")

print("\n" + "="*60)
print("ISSUE DETECTED:")
print("The model is predicting Autism=1.0 for all images!")
print("This suggests the model might be:")
print("  1. Overfitted or broken")
print("  2. Needs different preprocessing")
print("  3. The wrong model for this task")
print("="*60)
