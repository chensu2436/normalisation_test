"""
Copyright 2019 ETH Zurich, Seonwook Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#!/usr/bin/env python3

import os
import cv2
import numpy as np
from face import face
from KalmanFilter1D import Kalman1D
from normalisation import normalize
import scipy.io as sio
# from landmarks import landmarks


kalman_filters = list()
for point in range(2):
    # initialize kalman filters for different coordinates
    # will be used for face detection over a single object
    kalman_filters.append(Kalman1D(sz=100, R=0.01 ** 2))

# hard coded values from example
por = np.array([-127.790719, 4.621111, -12.025310]) # 3D gaze taraget position
hr = np.array([[-0.11660857],[0.14517431],[-0.07825662]])
ht = np.array([[11.53173266],[16.15314176],[431.66372868]])
head_pose = (hr, ht)

def detect_face(img):
    # detect face bounding box - use mtcnn, can be replaced by our own detector
    face_location = face.detect(img,  scale=0.25, use_max='SIZE')
    print("face bounding box:", face_location)

    if len(face_location) > 0:
        # use kalman filter to smooth bounding box position
        # assume work with complex numbers:
        output_tracked = kalman_filters[0].update(face_location[0] + 1j * face_location[1])
        face_location[0], face_location[1] = np.real(output_tracked), np.imag(output_tracked)
        output_tracked = kalman_filters[1].update(face_location[2] + 1j * face_location[3])
        face_location[2], face_location[3] = np.real(output_tracked), np.imag(output_tracked)

        # skip detecting facial points (in total 68 points)
    return face_location




def get_inputs_w_cam():
    print("reading camera param")
    cameraCalib = sio.loadmat('../data/calibration/cameraCalib.mat')
    camera_matrix = cameraCalib['cameraMatrix'] # shape (3,3)
    camera_distortion = cameraCalib['distCoeffs']
    fx, _, cx, _, fy, cy, _, _, _ = camera_matrix.flatten()
    camera_parameters = np.asarray([fx, fy, cx, cy])

    print("reading image")
    filepath = os.path.join('../data/example/day01_0087.jpg')
    img = cv2.imread(filepath)
    img = cv2.undistort(img, camera_matrix, camera_distortion)

    print("detecting face")
    face_location = detect_face(img)

    entry = {
            'full_frame': img,
            '3d_gaze_target': por,
            'camera_parameters': camera_parameters,
            'full_frame_size': (img.shape[0], img.shape[1]),
            'face_bounding_box': (int(face_location[0]), int(face_location[1]),
                                    int(face_location[2] - face_location[0]),
                                    int(face_location[3] - face_location[1]))
            }
    
    print("normalizing image")
    [patch, h_n, g_n, inverse_M, gaze_cam_origin, gaze_cam_target] = normalize(entry, head_pose)

    return [patch, h_n, g_n, inverse_M, gaze_cam_origin, gaze_cam_target]

