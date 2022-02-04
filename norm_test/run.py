import pickle
import torch
import os
import numpy as np
import cv2
import math
from test_with_cam_matrix import get_inputs_w_cam
from test_without_cam_matrix import get_inputs_wo_cam
import time

import warnings
warnings.filterwarnings("ignore")


def pitchyaw_to_vector(pitchyaw):
    vector = np.zeros((3, 1))
    vector[0, 0] = np.cos(pitchyaw[0]) * np.sin(pitchyaw[1])
    vector[1, 0] = np.sin(pitchyaw[0])
    vector[2, 0] = np.cos(pitchyaw[0]) * np.cos(pitchyaw[1])
    return vector

def mean_angle_loss(pred, truth):
    '''
    :param pred,truth: type=torch.Tensor
    :return:
    '''
    ans = 0
    for i in range(len(pred)):
        p_x, p_y, p_z = (pred[i][j] for j in range(3))
        t_x, t_y, t_z = (truth[i][j] for j in range(3))
        # print("p_x={}, p_y={}, p_z={}".format(p_x, p_y, p_z))
        # print("t_x={}, t_y={}, t_z={}".format(t_x, t_y, t_z))
        angles = (p_x * t_x + p_y * t_y + p_z * t_z)/(math.sqrt(p_x**2+p_y**2+p_z**2) * math.sqrt(t_x**2+t_y**2+t_z**2))
        ans += math.acos(angles) * 180 / np.pi
    return ans / len(pred)

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
    g_cnn = g_cnn
    # print("g_n: {} , g_cnn: {}".format(right_gaze, g_cnn))
    # print("g_n shape: {}, g_cnn shape: {}".format(right_gaze.shape, g_cnn.shape))
    
    return g_cnn


'''
Load demo weights
'''
start_time = time.time()
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
end_time_1 = time.time()
ted_weights = torch.load(ted_parameters_path)
if torch.cuda.device_count() == 1:
    if next(iter(ted_weights.keys())).startswith('module.'):
        ted_weights = dict([(k[7:], v) for k, v in ted_weights.items()])
gaze_network.load_state_dict(ted_weights)
end_time_2 = time.time()
print('finish loading model')

print('getting model: {}'.format(end_time_1 - start_time))
print('loading weights: {}'.format(end_time_2 - end_time_1))

# Test images
[patch1, h_n, g_n, inverse_M, gaze_cam_origin, gaze_cam_target] = get_inputs_w_cam()
start_time = time.time()
gaze1 = predict(gaze_network, patch1, h_n)
end_time = time.time()
# print("with adjusting to camera matrix:", gaze1)
# print("h_n:", h_n)
gaze_vector = pitchyaw_to_vector(g_n)
print("ground truth:", gaze_vector)
print("mean angle loss:", mean_angle_loss([gaze1], [gaze_vector]))
print('predicting 1 image: {}'.format(end_time - start_time))

# [patch2, h_n, g_n, inverse_M, gaze_cam_origin, gaze_cam_target] = get_inputs_wo_cam()
# gaze2 = predict(gaze_network, patch2, h_n)
# print("without ajusting to camera matrix", gaze2)
# print("h_n:", h_n)
# gaze_vector = pitchyaw_to_vector(g_n)
# print("ground truth:", gaze_vector)
# print("mean angle loss:", mean_angle_loss([gaze2], [gaze_vector]))
