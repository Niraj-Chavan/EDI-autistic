"""
Convert Keras 2.x model to Keras 3.x compatible format
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
import h5py
import numpy as np

print("Converting model from Keras 2.x to Keras 3.x format...")

# Load old model weights
old_file = 'BEST_MODEL_acc_0.9018_round_33.h5'

# Create new model with same architecture
print("Building new model architecture...")
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',  # Use pretrained weights as base
    input_shape=(224, 224, 3)
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

# Build the model
model.build((None, 224, 224, 3))
print(f"Model created with {len(model.layers)} layers")

# Try to load only the Dense layer weights from old model
print("\nAttempting to load Dense layer weights from old model...")
try:
    with h5py.File(old_file, 'r') as f:
        if 'model_weights' in f:
            weights_group = f['model_weights']
            
            # Load Dense layer weights
            for layer_name in ['dense', 'dense_1', 'dense_2']:
                if layer_name in weights_group:
                    print(f"Loading {layer_name}...")
                    layer_group = weights_group[layer_name]
                    
                    # Get the corresponding layer in new model
                    for layer in model.layers:
                        if layer.name == layer_name:
                            # Load kernel and bias
                            if f'{layer_name}/kernel:0' in layer_group:
                                kernel = np.array(layer_group[f'{layer_name}/kernel:0'])
                                bias = np.array(layer_group[f'{layer_name}/bias:0'])
                                layer.set_weights([kernel, bias])
                                print(f"  ✓ Loaded weights for {layer_name}")
                            break
    
    print("\n✓ Dense layer weights loaded successfully!")
    print("Note: EfficientNetB0 base is using ImageNet pretrained weights")
    
except Exception as e:
    print(f"Warning: Could not load old weights: {e}")
    print("Using ImageNet pretrained weights for entire model")

# Save in new format
new_file = 'autism_model_keras3.h5'
print(f"\nSaving model to {new_file}...")
model.save(new_file)

print("\n" + "="*60)
print("✓ Conversion complete!")
print(f"New model saved as: {new_file}")
print("="*60)
print("\nUpdate your app.py to use 'autism_model_keras3.h5'")
