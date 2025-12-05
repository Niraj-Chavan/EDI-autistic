"""
Properly convert the model by loading weights layer by layer
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
import h5py
import numpy as np

print("Proper model conversion...")

old_file = 'BEST_MODEL_acc_0.9018_round_33.h5'

# First, let's check what's in the old model
print("\nInspecting old model structure...")
with h5py.File(old_file, 'r') as f:
    print(f"Top-level keys: {list(f.keys())}")
    if 'model_weights' in f:
        print(f"Model weight groups: {list(f['model_weights'].keys())}")
        
        # Check EfficientNetB0 structure
        if 'efficientnetb0' in f['model_weights']:
            effnet = f['model_weights']['efficientnetb0']
            print(f"\nEfficientNetB0 has {len(list(effnet.keys()))} weight groups")
            
            # Count total weight arrays
            def count_weights(group):
                count = 0
                for key in group.keys():
                    if isinstance(group[key], h5py.Dataset):
                        count += 1
                    elif isinstance(group[key], h5py.Group):
                        count += count_weights(group[key])
                return count
            
            total = count_weights(effnet)
            print(f"Total weight arrays in EfficientNetB0: {total}")

# Now try to load the ENTIRE model using TF's legacy loader
print("\n\nAttempting to load with TF legacy format...")
try:
    # Try using the legacy h5 format loader
    from tensorflow.python.keras.saving import hdf5_format
    
    with h5py.File(old_file, 'r') as f:
        # Check if this is a full model save
        if 'model_config' in f.attrs:
            import json
            config = json.loads(f.attrs['model_config'])
            print(f"Model type: {config['class_name']}")
            print(f"Number of layers: {len(config['config']['layers'])}")
            
            # Try to reconstruct using the exact config
            model = tf.keras.models.model_from_json(json.dumps(config))
            print("Model structure recreated from config")
            
            # Now load weights
            print("Loading weights...")
            if 'model_weights' in f:
                # Use TF's internal weight loading
                hdf5_format.load_weights_from_hdf5_group(f['model_weights'], model.layers)
                print("✓ Weights loaded using TF internal loader!")
                
                # Save in new format
                new_file = 'autism_model_proper.h5'
                model.save(new_file)
                print(f"\n✓ Model saved as: {new_file}")
                
                # Test it
                print("\nTesting model...")
                test_img = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.float32)
                from tensorflow.keras.applications.efficientnet import preprocess_input
                test_img = preprocess_input(test_img)
                pred = model.predict(test_img, verbose=0)
                print(f"Test prediction: {pred[0][0]}")
                
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
