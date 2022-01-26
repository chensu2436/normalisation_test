import scipy.io as sio
import numpy as np
import sys

def get_cam_params():
    print("reading camera params")
    path = str(sys.argv[1])
    # print(path)
    cameraCalib = sio.loadmat(path)
    camera_matrix = cameraCalib['cameraMatrix'] # shape (3,3)
    camera_distortion = cameraCalib['distCoeffs']
    fx, _, cx, _, fy, cy, _, _, _ = camera_matrix.flatten()
    camera_parameters = np.asarray([fx, fy, cx, cy])
    # print(camera_matrix, camera_distortion, camera_parameters)
    return camera_matrix, camera_distortion, camera_parameters

camera_matrix, camera_distortion, camera_parameters = get_cam_params()

with open(str(sys.argv[2]), "w") as f:
    for line in camera_matrix:
        for num in line:
            f.write('{}\n'.format(num))

    f.write('\n')

    for line in camera_distortion:
        for num in line:
            f.write('{}\n'.format(num))

