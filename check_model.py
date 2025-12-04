import h5py
import json

f = h5py.File('BEST_MODEL_acc_0.9018_round_33.h5', 'r')
config = json.loads(f.attrs['model_config'])
layers = config['config']['layers']

print('Total layers:', len(layers))
print('\nLayer structure:')
for i, layer in enumerate(layers):
    print(f"{i}: {layer['class_name']} - {layer['config']['name']}")

print('\nModel weights groups:')
if 'model_weights' in f:
    for key in f['model_weights'].keys():
        print(f"  - {key}")

f.close()
