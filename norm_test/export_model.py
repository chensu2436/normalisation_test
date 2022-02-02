import torch
import numpy as np
import cv2

def preprocess_image(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    # cv2.imshow('processed patch', image)

    image = np.transpose(image, [2, 0, 1])  # CxHxW
    image = 2.0 * image / 255.0 - 1
    return image

'''
Load demo weights
'''
ted_parameters_path = 'demo_weights/weights_ted.pth.tar'
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

gaze_network.eval()

image = cv2.imread("../normalisation_test_data/p00/01/norm.png")
processed_patch = preprocess_image(image)
processed_patch = processed_patch[np.newaxis, :, :, :]
input_dict = {
            'image_a': processed_patch,
            'gaze_a': [],
            'head_a': [1, 1],
            'R_gaze_a': [],
            'R_head_a': [],
    }

torch.onnx.export(gaze_network,         # model being run 
    input_dict,       # model input (or a tuple for multiple inputs) 
    "gaze_model.onnx",       # where to save the model  
    export_params=True,  # store the trained parameter weights inside the model file 
    opset_version=10,    # the ONNX version to export the model to 
    do_constant_folding=True,  # whether to execute constant folding for optimization 
    input_names = ['modelInput'],   # the model's input names 
    output_names = ['modelOutput'], # the model's output names 
)
print(" ") 
print('Model has been converted to ONNX') 


# model_scripted = torch.jit.script(gaze_network) # Export to TorchScript
# model_scripted.save('gaze_model_scripted.pt') # Save
