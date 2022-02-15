import numpy as np
import math
import cv2
import sys

def read_rvec(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    f.close()
    rvec = np.asarray([float(lines[1]), float(lines[2]), float(lines[3])]).reshape(3, 1)
    tvec = np.asarray([float(lines[5]), float(lines[6]), float(lines[7])]).reshape(3, 1)
    return rvec, tvec


'''
https://learnopencv.com/rotation-matrix-to-euler-angles/
'''

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2]) # roll
        y = math.atan2(-R[2, 0], sy) # pitch
        z = math.atan2(R[1, 0], R[0, 0]) # yaw
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([z, y, x])

'''
https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV/blob/master/pose_estimation.py
'''

def convert(rvec_matrix, translation_vector):
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    return np.array([yaw, pitch, roll])

rvec, tvec = read_rvec(sys.argv[1])
# print(rvec)
rotate_mat, _ = cv2.Rodrigues(rvec)
ypr_d = convert(rotate_mat, tvec)
print(ypr_d)
# ypr_r = rotationMatrixToEulerAngles(rotate_mat)
# print(ypr_r * 180 /math.pi)