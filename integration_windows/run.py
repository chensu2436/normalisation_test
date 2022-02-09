import torch
import os
import numpy as np
import cv2
import math
import sys
import pandas as pd

from models import DTED

# Functions to calculate relative rotation matrices for gaze dir. and head pose
def R_x(theta):
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    return np.array([
        [1., 0., 0.],
        [0., cos_, -sin_],
        [0., sin_, cos_]
    ]).astype(np.float32)

def R_y(phi):
    sin_ = np.sin(phi)
    cos_ = np.cos(phi)
    return np.array([
        [cos_, 0., sin_],
        [0., 1., 0.],
        [-sin_, 0., cos_]
    ]).astype(np.float32)

def calculate_rotation_matrix(e):
    return np.matmul(R_y(e[1]), R_x(e[0]))

def preprocess_image(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    image = np.transpose(image, [2, 0, 1])  # CxHxW
    image = 2.0 * image / 255.0 - 1
    return image

def predict(gaze_network, image, head_pose):
    processed_patch = preprocess_image(image)
    processed_patch = processed_patch[np.newaxis, :, :, :]
    R_head_a = calculate_rotation_matrix(head_pose)
    R_gaze_a = np.zeros((1, 3, 3))
    input_dict = {
            'image_a': processed_patch,
            'gaze_a': [],
            'head_a': head_pose,
            'R_gaze_a': R_gaze_a,
            'R_head_a': R_head_a,
    }

    # compute eye gaze
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
    
    return g_cnn

def init_model():
    ted_parameters_path = 'demo_weights/weights_ted.pth.tar'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    return gaze_network


if __name__ == "__main__":
    gaze_model = init_model()
    dir_name = sys.argv[1]
    csv_file = dir_name + "/test.csv"
    df = pd.read_csv(csv_file)
    gaze_x = []
    gaze_y = []
    gaze_z = []

    for i in df.index:
        image = cv2.imread(dir_name + "/" + df['image_file'][i])
        head_pose = (float(df['h_x'][i]), float(df['h_y'][i]))
        gaze = predict(gaze_model, image, head_pose)
        gaze_x.append(gaze[0])
        gaze_y.append(gaze[1])
        gaze_z.append(gaze[2])

    
    df['gaze_x'] = gaze_x
    df['gaze_y'] = gaze_y
    df['gaze_z'] = gaze_z
    df.to_csv(csv_file, index=False)
        


