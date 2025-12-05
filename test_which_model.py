import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

print("Testing model loading...")
print(f"TensorFlow version: {tf.__version__}")

# Test 1: Legacy loader
try:
    from tensorflow.python.keras.saving import hdf5_format
    import h5py
    
    with h5py.File('BEST_MODEL_acc_0.9018_round_33.h5', 'r') as f:
        model = hdf5_format.load_model_from_hdf5(f)
    print("✓ Original model loaded with legacy loader!")
    print(f"Model layers: {len(model.layers)}")
except Exception as e:
    print(f"✗ Legacy loader failed: {str(e)[:100]}")
    
    # Test 2: Standard loader
    try:
        model = tf.keras.models.load_model('BEST_MODEL_acc_0.9018_round_33.h5', compile=False)
        print("✓ Original model loaded with standard loader!")
        print(f"Model layers: {len(model.layers)}")
    except Exception as e2:
        print(f"✗ Standard loader failed: {str(e2)[:100]}")
        
        # Test 3: Demo model
        try:
            model = tf.keras.models.load_model('autism_model_demo.h5', compile=False)
            print("✓ Demo model loaded (random weights)")
            print(f"Model layers: {len(model.layers)}")
        except Exception as e3:
            print(f"✗ Demo model failed: {str(e3)[:100]}")

# Test prediction
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

test_img = np.random.randint(0, 255, (1, 224, 224, 3)).astype(np.float32)
test_img = preprocess_input(test_img)

pred = model.predict(test_img, verbose=0)
print(f"\nTest prediction: {pred[0][0]:.6f}")
