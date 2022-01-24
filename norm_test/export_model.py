import torch

'''
Load demo weights
'''
ted_parameters_path = '../demo_weights/weights_ted.pth.tar'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from models import DTED
gaze_network = DTED(
    growth_rate=32,
    z_dim_app=64,
    z_dim_gaze=2,
    z_dim_head=16,
    decoder_input_c=32,
    normalize_3d_codes=True,
    normalize_3d_codes_axis=1,
    backprop_gaze_to_encoder=False,
).to(device)

ted_weights = torch.load(ted_parameters_path)
if torch.cuda.device_count() == 1:
    if next(iter(ted_weights.keys())).startswith('module.'):
        ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])
gaze_network.load_state_dict(ted_weights)

print('finish loading model')

model_scripted = torch.jit.script(gaze_network) # Export to TorchScript
model_scripted.save('gaze_model_scripted.pt') # Save
