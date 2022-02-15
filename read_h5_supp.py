import h5py
import numpy as np
filename = "MPIIFaceGaze_supplementary.h5"

with h5py.File(filename, "r") as f:
    # List all groups
    # print("Keys: %s" % f.keys())
    # a_group_key = list(f.keys())[0]

    # # Get the data
    # data = list(f[a_group_key])
    pid, group = list(f.items())[0]
    print(pid, group)
    print("Keys: %s" % group.keys())
    a_group_key = list(group.keys())[0]
    num_entries = next(iter(group.values())).shape[0]
    print(num_entries)
    # fx, fy, cx, cy = group['camera_parameters'][i, :]
    # distortion = group['distortion_parameters'][i, :],
    # for i in range(num_entries):
    #     fx, fy, cx, cy = group['camera_parameters'][i, :]
    #     camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    #                          dtype=np.float64)
    #     distortion = group['distortion_parameters'][i, :]
    #     print(camera_matrix, distortion)
    # for i in range(num_entries):
    #     head_pose = group['head_pose'][i, 3:]
    #     print(head_pose)
    for i in range(num_entries):
        gaze = group['3d_gaze_target'][i, :]
        filename = group['file_name'][i]
        print(gaze, filename)
    