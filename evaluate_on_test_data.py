import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("EVALUATING MODEL ON TEST DATA")
print("="*60)

# Load model
model = tf.keras.models.load_model('BEST_MODEL_acc_0.9518_round_33.h5', compile=False)
print("Model loaded successfully!")

def preprocess_image(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize and preprocess
    img_resized = cv2.resize(img_array, (224, 224))
    img_batch = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    
    return img_preprocessed

# Test on autistic images
print("\n1. TESTING ON AUTISTIC IMAGES (should predict Autism):")
autistic_dir = 'test/autistic'
autistic_files = [f for f in os.listdir(autistic_dir) if f.endswith('.jpg')][:20]  # Test first 20

correct_autistic = 0
for img_file in autistic_files:
    img_path = os.path.join(autistic_dir, img_file)
    try:
        processed = preprocess_image(img_path)
        pred = model.predict(processed, verbose=0)[0][0]
        
        # Using threshold 0.35
        result = "Autism" if pred > 0.35 else "Non-Autism"
        is_correct = result == "Autism"
        if is_correct:
            correct_autistic += 1
        
        status = "✓" if is_correct else "✗"
        print(f"  {status} {img_file}: {result} (prob: {pred:.4f})")
    except Exception as e:
        print(f"  ⚠ {img_file}: Skipped (corrupted)")

print(f"\nAutistic accuracy: {correct_autistic}/{len(autistic_files)} = {correct_autistic/len(autistic_files)*100:.1f}%")

# Test on non-autistic images
print("\n2. TESTING ON NON-AUTISTIC IMAGES (should predict Non-Autism):")
non_autistic_dir = 'test/non-autistic'
non_autistic_files = [f for f in os.listdir(non_autistic_dir) if f.endswith('.jpg')][:20]  # Test first 20

correct_non_autistic = 0
for img_file in non_autistic_files:
    img_path = os.path.join(non_autistic_dir, img_file)
    try:
        processed = preprocess_image(img_path)
        pred = model.predict(processed, verbose=0)[0][0]
        
        # Using threshold 0.35
        result = "Autism" if pred > 0.35 else "Non-Autism"
        is_correct = result == "Non-Autism"
        if is_correct:
            correct_non_autistic += 1
        
        status = "✓" if is_correct else "✗"
        print(f"  {status} {img_file}: {result} (prob: {pred:.4f})")
    except Exception as e:
        print(f"  ⚠ {img_file}: Skipped (corrupted)")

print(f"\nNon-Autistic accuracy: {correct_non_autistic}/{len(non_autistic_files)} = {correct_non_autistic/len(non_autistic_files)*100:.1f}%")

# Overall accuracy
total_correct = correct_autistic + correct_non_autistic
total_tested = len(autistic_files) + len(non_autistic_files)
overall_accuracy = total_correct / total_tested * 100

print("\n" + "="*60)
print(f"OVERALL ACCURACY: {total_correct}/{total_tested} = {overall_accuracy:.1f}%")
print("="*60)

# Recommendation
if overall_accuracy < 60:
    print("\n⚠️ RECOMMENDATION: Threshold needs adjustment")
    print("   Try different threshold values (0.2, 0.3, 0.4, 0.5)")
elif overall_accuracy < 80:
    print("\n✓ Model is working but could be better")
    print("   Consider fine-tuning the threshold")
else:
    print("\n✓ Model is performing well!")
