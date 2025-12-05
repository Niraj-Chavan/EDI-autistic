import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.densenet import preprocess_input
import warnings
warnings.filterwarnings('ignore')

print("Testing with DenseNet preprocessing...")

model = tf.keras.models.load_model('87.00.h5', compile=False)

print("\nTesting with 10 random images:")
for i in range(10):
    test_img = np.random.randint(0, 255, (1, 224, 224, 3)).astype(np.float32)
    test_img_preprocessed = preprocess_input(test_img)
    
    pred = model.predict(test_img_preprocessed, verbose=0)
    
    non_autism = pred[0][0]
    autism = pred[0][1]
    result = "Autism" if autism > non_autism else "Non-Autism"
    conf = max(autism, non_autism) * 100
    print(f"  {i+1}. {result} ({conf:.1f}%) - Autism: {autism:.4f}, Non-Autism: {non_autism:.4f}")

print("\n✓ Now testing with DenseNet preprocessing!")
