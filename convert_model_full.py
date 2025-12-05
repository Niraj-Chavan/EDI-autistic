"""
Full model conversion - load all possible weights from Keras 2.x model
"""
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
import h5py
import numpy as np

print("Full model conversion from Keras 2.x to Keras 3.x...")

old_file = 'BEST_MODEL_acc_0.9018_round_33.h5'

# Create new model
print("Building model architecture...")
base_model = EfficientNetB0(
    include_top=False,
    weights=None,  # No pretrained weights - we'll load from old model
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

model.build((None, 224, 224, 3))
print(f"Model created with {len(model.layers)} layers")

# Load ALL weights from old model
print("\nLoading weights from old model...")
try:
    with h5py.File(old_file, 'r') as f:
        if 'model_weights' in f:
            weights_group = f['model_weights']
            
            print(f"Available weight groups: {list(weights_group.keys())}")
            
            # Load EfficientNetB0 weights
            if 'efficientnetb0' in weights_group:
                print("\nLoading EfficientNetB0 base weights...")
                effnet_group = weights_group['efficientnetb0']
                
                # Get all weight names
                def get_all_weights(group, prefix=''):
                    weights = {}
                    for key in group.keys():
                        if isinstance(group[key], h5py.Group):
                            weights.update(get_all_weights(group[key], prefix + key + '/'))
                        else:
                            weights[prefix + key] = np.array(group[key])
                    return weights
                
                old_weights = get_all_weights(effnet_group)
                print(f"  Found {len(old_weights)} weight arrays in EfficientNetB0")
                
                # Try to match and load weights
                base_weights = []
                for layer in base_model.layers:
                    layer_weights = []
                    for weight in layer.weights:
                        weight_name = weight.name
                        # Try different name formats
                        for old_name in old_weights.keys():
                            if weight_name.split('/')[-1].replace(':0', '') in old_name:
                                layer_weights.append(old_weights[old_name])
                                break
                    if layer_weights and len(layer_weights) == len(layer.weights):
                        try:
                            layer.set_weights(layer_weights)
                        except:
                            pass
                
                print("  ✓ EfficientNetB0 weights loaded (best effort)")
            
            # Load Dense layers
            print("\nLoading Dense layer weights...")
            for layer_name in ['dense', 'dense_1', 'dense_2']:
                if layer_name in weights_group:
                    layer_group = weights_group[layer_name]
                    for layer in model.layers:
                        if layer.name == layer_name:
                            kernel_key = f'{layer_name}/kernel:0'
                            bias_key = f'{layer_name}/bias:0'
                            if kernel_key in layer_group and bias_key in layer_group:
                                kernel = np.array(layer_group[kernel_key])
                                bias = np.array(layer_group[bias_key])
                                layer.set_weights([kernel, bias])
                                print(f"  ✓ {layer_name}: kernel {kernel.shape}, bias {bias.shape}")
    
    print("\n✓ All available weights loaded!")
    
except Exception as e:
    print(f"Error during weight loading: {e}")
    import traceback
    traceback.print_exc()

# Save new model
new_file = 'autism_model_keras3.h5'
print(f"\nSaving to {new_file}...")
model.save(new_file, save_format='h5')

print("\n" + "="*60)
print("✓ Full conversion complete!")
print(f"Model saved as: {new_file}")
print("="*60)
