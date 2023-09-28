import numpy as np

CMU0_Path = 'logs/panoptic_160906_pizza1_CMU0_poses3ds.npy'

pose3d = np.load(CMU0_Path,allow_pickle=True)

seq = '160906_pizza1'

# for seq in self.sequence_list: 
    # for a specific dataset
# cameras = self._get_cam(seq)
# cam_num = len(cameras)
# curr_anno = osp.join(self.dataset_root,
#                         seq, 'hdPose3d_stage1_coco19')
# anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))

# seq_count[seq] = 0
# for i, file in enumerate(anno_files):
#     # one frame of different cameras
#     if i % self._interval == 0:
#         with open(file) as dfile:
#             bodies = json.load(dfile)['bodies']
#         if len(bodies) == 0:
#             continue
        
#         # check situation of different cameras
#         all_people_observable = []
#         for k, v in cameras.items():
#             postfix = osp.basename(file).replace('body3DScene', '')
#             prefix = '{:02d}_{:02d}'.format(k[0], k[1])
#             image = osp.join(seq, 'hdImgs', prefix,
#                                 prefix + postfix)
#             image = image.replace('json', 'jpg')

#             all_poses_3d = []
#             all_poses_vis_3d = []
#             all_poses = []
#             all_poses_vis = []
#             for body in bodies:
#                 pose3d = np.array(body['joints19'])\
#                     .reshape((-1, 4))
#                 pose3d = pose3d[:self.num_joints]

#                 joints_vis = pose3d[:, -1] > 0.1

#                 if not joints_vis[self.root_id]:
#                     continue

#                 # Coordinate transformation
#                 M = np.array([[1.0, 0.0, 0.0],
#                                 [0.0, 0.0, -1.0],
#                                 [0.0, 1.0, 0.0]])
#                 pose3d[:, 0:3] = pose3d[:, 0:3].dot(M)

#                 all_poses_3d.append(pose3d[:, 0:3] * 10.0)
#                 all_poses_vis_3d.append(
#                     np.repeat(
#                         np.reshape(
#                             joints_vis, (-1, 1)), 3, axis=1))

#                 pose2d = np.zeros((pose3d.shape[0], 2))
#                 pose2d[:, :2] = projectPoints(
#                     pose3d[:, 0:3].transpose(), v['K'], v['R'],
#                     v['t'], v['distCoef']).transpose()[:, :2]
#                 x_check = \
#                     np.bitwise_and(pose2d[:, 0] >= 0,
#                                     pose2d[:, 0] <= width - 1)
#                 y_check = \
#                     np.bitwise_and(pose2d[:, 1] >= 0,
#                                     pose2d[:, 1] <= height - 1)
#                 check = np.bitwise_and(x_check, y_check)
#                 joints_vis[np.logical_not(check)] = 0

#                 all_poses.append(pose2d)
#                 all_poses_vis.append(
#                     np.repeat(
#                         np.reshape(
#                             joints_vis, (-1, 1)), 2, axis=1))

#             all_people_observable.append(all_poses_vis)
#             # check if there are any false
#             # for this camera, can all the bodies be visible?
#             # all_observed = True
#             # for arr in all_poses_vis:
#                 # fail_pos = np.where(arr.reshape(-1)==False)
#                 # if len(fail_pos) > 0:
#                     # all_observed = False
#                     # break

#             if len(all_poses_3d) > 0:
#                 our_cam = {}
#                 our_cam['R'] = v['R']
#                 our_cam['T'] = -np.dot(
#                     v['R'].T, v['t']) * 10.0  # cm to mm
#                 our_cam['standard_T'] = v['t'] * 10.0
#                 our_cam['fx'] = np.array(v['K'][0, 0])
#                 our_cam['fy'] = np.array(v['K'][1, 1])
#                 our_cam['cx'] = np.array(v['K'][0, 2])
#                 our_cam['cy'] = np.array(v['K'][1, 2])
#                 our_cam['k'] = v['distCoef'][[0, 1, 4]]\
#                     .reshape(3, 1)
#                 our_cam['p'] = v['distCoef'][[2, 3]]\
#                     .reshape(2, 1)

#                 db.append({
#                     'key': "{}_{}{}".format(
#                         seq, prefix, postfix.split('.')[0]),
#                     'image': osp.join(self.dataset_root, image),
#                     'joints_3d': all_poses_3d,
#                     'joints_3d_vis': all_poses_vis_3d,
#                     'joints_2d': all_poses,
#                     'joints_2d_vis': all_poses_vis,
#                     'camera': our_cam
#                 })