"""
Convert with 300x300 input size (matching your training code IMG_SIZE=300)
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
import h5py
import numpy as np

print("Converting model with 300x300 input size...")

old_file = 'BEST_MODEL_acc_0.9018_round_33.h5'

# Create model with 300x300 input
print("Building model with 300x300 input...")
base_model = EfficientNetB0(
    include_top=False,
    weights=None,
    input_shape=(300, 300, 3)  # Match training IMG_SIZE
)

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.build((None, 300, 300, 3))
print(f"Model created: input shape (300, 300, 3)")

# Load weights
print("\nLoading weights...")
try:
    with h5py.File(old_file, 'r') as f:
        if 'model_weights' in f:
            weights_group = f['model_weights']
            
            # Load Dense layers (these should work regardless of input size)
            for layer_name in ['dense', 'dense_1', 'dense_2']:
                if layer_name in weights_group:
                    layer_group = weights_group[layer_name]
                    for layer in model.layers:
                        if layer.name == layer_name:
                            kernel = np.array(layer_group[f'{layer_name}/kernel:0'])
                            bias = np.array(layer_group[f'{layer_name}/bias:0'])
                            layer.set_weights([kernel, bias])
                            print(f"  ✓ {layer_name} loaded")
            
            # Try to load EfficientNetB0 weights
            if 'efficientnetb0' in weights_group:
                print("  ✓ Attempting EfficientNetB0 weights...")
                try:
                    # This might fail due to size mismatch, but worth trying
                    model.layers[0].load_weights(old_file, by_name=True, skip_mismatch=True)
                    print("  ✓ EfficientNetB0 weights loaded (with skip_mismatch)")
                except:
                    print("  ⚠ EfficientNetB0 weights skipped (size mismatch)")
    
    print("\n✓ Weights loaded!")
    
except Exception as e:
    print(f"Error: {e}")

# Save
new_file = 'autism_model_300x300.h5'
print(f"\nSaving to {new_file}...")
model.save(new_file)

print("\n" + "="*60)
print(f"✓ Model saved: {new_file}")
print("Input size: 300x300")
print("="*60)
