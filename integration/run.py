import scipy.io as sio
import os
import cv2
import numpy as np

def get_cam_params():
    print("reading camera params")
    cameraCalib = sio.loadmat('../data/calibration/cameraCalib.mat')
    camera_matrix = cameraCalib['cameraMatrix'] # shape (3,3)
    camera_distortion = cameraCalib['distCoeffs']
    fx, _, cx, _, fy, cy, _, _, _ = camera_matrix.flatten()
    camera_parameters = np.asarray([fx, fy, cx, cy])
    return camera_matrix, camera_distortion, camera_parameters

def get_undistorted_image(camera_matrix, camera_distortion):
    print("reading image")
    filepath = os.path.join('../data/example/day01_0087.jpg')
    img = cv2.imread(filepath)
    img = cv2.undistort(img, camera_matrix, camera_distortion)
    return img

def load_face_model():
    print("loading face model")
    return sio.loadmat('../data/faceModelGeneric.mat')['model']

def get_face_points_bbox():
    print("loading face points")
    with open("../output.txt") as f:
        lines = f.readlines()

    x = lines[1].split()
    y = lines[2].split()

    points = []
    indices = [19, 22, 25, 28, 31, 37] # four eye corners and two mouth corners
    for i in indices:
        points.append((int(float(x[i])),int(float(y[i]))))

    bbox = [int(x) for x in lines[4].split()]
    return np.asarray(points), np.asarray(bbox)

def get_head_pose(face, points, camera_matrix, camera_distortion):
    print("estimating head pose")
    num_pts = face.shape[1]
    facePts = face.T.reshape(num_pts, 1, 3)
    landmarks = points.astype(np.float32)
    landmarks = landmarks.reshape(num_pts, 1, 2)
    ret, rvec, tvec = cv2.solvePnP(facePts, landmarks, camera_matrix, camera_distortion, flags=cv2.SOLVEPNP_EPNP)
    
    ## further optimize
    ret, rvec, tvec = cv2.solvePnP(facePts, landmarks, camera_matrix, camera_distortion, rvec, tvec, True)

    return rvec, tvec

def common_pre(entry, head_pose):
    print("doing common_pre")
    rvec, tvec = head_pose
    if rvec is None or tvec is None:
        raise ValueError('rvec or tvec is None')

    # Calculate rotation matrix and euler angles
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)
    rotate_mat, _ = cv2.Rodrigues(rvec)

    # Reconstruct frame
    full_frame = cv2.cvtColor(entry['full_frame'], cv2.COLOR_BGR2RGB)

    # Form camera matrix
    fx, fy, cx, cy = entry['camera_parameters']
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                             dtype=np.float64)

    # Get camera parameters
    normalized_parameters = {
        'focal_length': 1300,
        'distance': 600,
        'size': (256, 64),
    }
    n_f = normalized_parameters['focal_length']
    n_d = normalized_parameters['distance']
    ow, oh = normalized_parameters['size']
    norm_camera_matrix = np.array([[n_f, 0, 0.5*ow], [0, n_f, 0.5*oh], [0, 0, 1]],
                                  dtype=np.float64)

    ## compute estimated 3D positions of the gaze origin
    face =  load_face_model()
    Fc = np.dot(rotate_mat, face) + tvec # 3D positions of facial landmarks
    re = 0.5*(Fc[:,0] + Fc[:,1]).reshape((3,1)) # center of left eye
    le = 0.5*(Fc[:,2] + Fc[:,3]).reshape((3,1)) # center of right eye
    g_o = (re + le) / 2 # center of two eyes
    return [full_frame, rotate_mat, camera_matrix, n_d,
            norm_camera_matrix, ow, oh, g_o] 

def normalize(entry, head_pose):
    [full_frame, rotate_mat, camera_matrix, n_d, norm_camera_matrix,
     ow, oh, g_o] = common_pre(entry, head_pose)
    
    # Code below is an adaptation of code by Xucong Zhang
    # https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/revisiting-data-normalization-for-appearance-based-gaze-estimation/

    print("normalizing eye patch")
    distance = np.linalg.norm(g_o)
    z_scale = n_d / distance
    S = np.eye(3, dtype=np.float64)
    S[2, 2] = z_scale

    hRx = rotate_mat[:, 0]
    forward = (g_o / np.linalg.norm(g_o)).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T # rotation matrix R

    W = np.dot(np.dot(norm_camera_matrix, S),
               np.dot(R, np.linalg.inv(camera_matrix))) # transformation matrix
    patch = cv2.warpPerspective(full_frame, W, (ow, oh)) # image normalization

    R = np.asmatrix(R)

    # Correct head pose
    head_mat = R * rotate_mat
    n_h = np.array([np.arcsin(head_mat[1, 2]), np.arctan2(head_mat[0, 2], head_mat[2, 2])])

    return patch, n_h



camera_matrix, camera_distortion, camera_parameters = get_cam_params()
img = get_undistorted_image(camera_matrix, camera_distortion)
face = load_face_model()
face_points, bbox = get_face_points_bbox()
hr, ht = get_head_pose(face, face_points, camera_matrix, camera_distortion)

print("hr={}, ht={}".format(hr, ht))

entry = {
            'full_frame': img,
            'camera_parameters': camera_parameters,
            }

patch, n_h = normalize(entry, (hr, ht))
patch_rgb = cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)
cv2.imshow('normalized_eye_patch', patch_rgb)
cv2.waitKey(3000)
cv2.destroyAllWindows()