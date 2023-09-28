import json
import argparse
import pickle
import numpy as np

CAM_LIST={
    'CMU0_ori': [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)],  # Origin order in MvP
    'CMU0' : [(0, 3), (0, 6),(0, 12),(0, 13), (0, 23)],
    'CMU0-3' : [(0, 3), (0, 6),(0, 12)],
    'CMU0-4' : [(0, 3), (0, 6),(0, 12),(0, 13)],
    'CMU0-6' : [(0, 3), (0, 6),(0, 12),(0, 13), (0, 23), (0, 10)],
    'CMU0-7' : [(0, 3), (0, 6),(0, 12),(0, 13), (0, 23), (0, 10), (0, 16)],
    'CMU0ex' : [(0, 3), (0, 6), (0, 12),(0, 13), (0, 23), (0, 10), (0, 16)],
    'CMU1' : [(0, 1),(0, 2),(0, 3),(0, 4),(0, 6),(0, 7),(0, 10)],  
    'CMU2' : [(0, 12), (0, 16), (0, 18), (0, 19), (0, 22), (0, 23), (0, 30)],
    'CMU3': [(0, 10), (0, 12), (0, 16), (0, 18)],
    'CMU4' : [(0, 6), (0, 7), (0, 10), (0, 12), (0, 16), (0, 18), (0, 19), (0, 22), (0, 23), (0, 30)],
}
parser = argparse.ArgumentParser ()
parser.add_argument('--seq', type=str, help='A string argument')
parser.add_argument('--cam', type=str, help='A string argument')
args = parser.parse_args ()
seq, cam = args.seq, args.cam

json_file_path = f'datasets/panoptic/{seq}/calibration_{seq}.json'
cam_list = CAM_LIST[cam]

with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)
    cameras = data['cameras']
    selected_cameras = []
    for cam_id, cam_name in enumerate(cam_list):
        for camera in cameras:
            if camera['name'] == f'{cam_name[0]:02}_{cam_name[1]:02}':
                selected_cameras.append(camera)
                break

print(selected_cameras)

K = np.stack([np.array(camera['K']) for camera in selected_cameras], axis=0).astype(np.float32)
# Rt = np.hstack((R, t.reshape(-1, 1)))
T = np.stack([np.array(camera['t']) for camera in selected_cameras], axis=0).astype(np.float32)
R = np.stack([np.array(camera['R']) for camera in selected_cameras], axis=0).astype(np.float32)
T = np.stack([ - np.dot(R[i], T[i]) for i in range(len(selected_cameras))], axis=0).astype(np.float64)

RT_pre = np.stack([np.hstack((np.array(camera['R']), np.array(camera['t']).reshape(-1, 1))) for camera in selected_cameras], axis=0).astype(np.float32)
RT = np.stack([ np.concatenate((R[i], T[i]),axis=-1) for i in range(len(selected_cameras))], axis=0).astype(np.float64)


P = np.stack([np.dot(K[i], RT[i]) for i in range(len(selected_cameras))], axis=0).astype(np.float64)

camera_parameter = {
    'K': K,
    'RT': RT,
    'P': P,
}

with open ( "datasets/CampusSeq1/camera_parameter.pickle",'rb' ) as f:
    std_camera_parameter = pickle.load (f)

# 现在，变量"data"包含了JSON文件中的数据，可以按需使用它
print(data)