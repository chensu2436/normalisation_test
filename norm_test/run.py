import pickle
import torch
import os
import numpy as np
import cv2
from test_with_cam_matrix import get_inputs_w_cam
from test_without_cam_matrix import get_inputs_wo_cam

def preprocess_image(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    # cv2.imshow('processed patch', image)

    image = np.transpose(image, [2, 0, 1])  # CxHxW
    image = 2.0 * image / 255.0 - 1
    return image

def predict(gaze_network, image, head_pose):
    processed_patch = preprocess_image(image)
    processed_patch = processed_patch[np.newaxis, :, :, :]
    # print("patch shape: {}".format(patch.shape))
    input_dict = {
            'image_a': processed_patch,
            'gaze_a': [],
            'head_a': head_pose,
            'R_gaze_a': [],
            'R_head_a': [],
    }

    # compute eye gaze and point of regard
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for k, v in input_dict.items():
        input_dict[k] = torch.FloatTensor(v).to(device).detach()

    gaze_network.eval()
    output_dict = gaze_network(input_dict)
    output_dict = dict([(k, v.cpu().detach().numpy()) for k, v in output_dict.items()])
    output = output_dict['gaze_a_hat']
    g_cnn = output
    g_cnn = g_cnn.reshape(3, 1)
    g_cnn /= np.linalg.norm(g_cnn)
    g_cnn = -g_cnn
    # print("g_n: {} , g_cnn: {}".format(right_gaze, g_cnn))
    # print("g_n shape: {}, g_cnn shape: {}".format(right_gaze.shape, g_cnn.shape))
    
    return g_cnn


#################################
# Load gaze network
#################################
ted_parameters_path = 'demo_weights/weights_ted.pth.tar'
maml_parameters_path = 'demo_weights/weights_maml'
k = 9

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create network
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

#################################

# Load T-ED weights if available
assert os.path.isfile(ted_parameters_path)
print('> Loading: %s' % ted_parameters_path)
ted_weights = torch.load(ted_parameters_path)
if torch.cuda.device_count() == 1:
    if next(iter(ted_weights.keys())).startswith('module.'):
        ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])


gaze_network.load_state_dict(ted_weights)
print('finish loading model')

#####################################


# Test images
[patch1, h_n, g_n, inverse_M, gaze_cam_origin, gaze_cam_target] = get_inputs_w_cam()
gaze1 = predict(gaze_network, patch1, h_n)
print(gaze1)

[patch2, h_n, g_n, inverse_M, gaze_cam_origin, gaze_cam_target] = get_inputs_wo_cam()
gaze2 = predict(gaze_network, patch2, h_n)
print(gaze2)
