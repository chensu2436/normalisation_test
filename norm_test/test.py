import numpy as np
# import scipy.io as sio
# from KalmanFilter1D import Kalman1D
# from face import face
# import cv2
# import os


# kalman_filters = list()
# for point in range(2):
#     # initialize kalman filters for different coordinates
#     # will be used for face detection over a single object
#     kalman_filters.append(Kalman1D(sz=100, R=0.01 ** 2))


# # hard coded values from example
# por = np.array([-127.790719, 4.621111, -12.025310]) # 3D gaze taraget position
# hr = np.array([[-0.11660857],[0.14517431],[-0.07825662]])
# ht = np.array([[11.53173266],[16.15314176],[431.66372868]])
# head_pose = (hr, ht)

# def detect_face(img):
#     # detect face bounding box - use mtcnn, can be replaced by our own detector
#     face_location = face.detect(img,  scale=0.25, use_max='SIZE')
#     print("face bounding box:", face_location)

#     if len(face_location) > 0:
#         # use kalman filter to smooth bounding box position
#         # assume work with complex numbers:
#         output_tracked = kalman_filters[0].update(face_location[0] + 1j * face_location[1])
#         face_location[0], face_location[1] = np.real(output_tracked), np.imag(output_tracked)
#         output_tracked = kalman_filters[1].update(face_location[2] + 1j * face_location[3])
#         face_location[2], face_location[3] = np.real(output_tracked), np.imag(output_tracked)

#         # skip detecting facial points (in total 68 points)
#     return face_location


# print("reading camera param")
# cameraCalib = sio.loadmat('../data/calibration/cameraCalib.mat')
# camera_matrix = cameraCalib['cameraMatrix'] # shape (3,3)
# camera_distortion = cameraCalib['distCoeffs']
# fx, _, cx, _, fy, cy, _, _, _ = camera_matrix.flatten()
# camera_parameters = np.asarray([fx, fy, cx, cy])

# print("reading image")
# filepath = os.path.join('../data/example/day01_0087.jpg')
# img = cv2.imread(filepath)
# img = cv2.undistort(img, camera_matrix, camera_distortion)

# print("detecting face")
# face_location = detect_face(img)
# print(face_location)

def vector_to_pitchyaw(vectors):
    """Convert given gaze vectors to yaw (theta) and pitch (phi) angles."""
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


hr = np.array([[-0.11660857],[0.14517431],[-0.07825662]]).T
print(vector_to_pitchyaw(hr))