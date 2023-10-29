
from torch.utils.data import Dataset
import numpy as np
import re
from glob import glob
import pickle
import os
import os.path as osp
from collections import OrderedDict
import cv2
from prettytable import PrettyTable
import json
from tqdm import tqdm

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
# cam_list = list(set(CAM_LIST['CMU0ex'] + CAM_LIST['CMU1'] + CAM_LIST['CMU2'] + CAM_LIST['CMU3'] + CAM_LIST['CMU4']))
seq_list = ['160906_pizza1', '160422_haggling1', '160906_ian5', '160906_band4'],
interval = 12
ROOTIDX = 2
CMU_JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}

COCO_JOINTS_DEF = {
    "nose": 0,
    "l-eye": 1,
    "r-eye": 2,
    "l-ear": 3,
    "r-ear": 4,
    "l-shoulder": 5,
    "r-shoulder": 6,
    "l-elbow": 7,
    "r-elbow": 8,
    "l-wrist": 9,
    "r-wrist": 10,
    "l-hip": 11,
    "r-hip": 12,
    "l-knee": 13,
    "r-knee": 14,
    "l-ankle": 15,
    "r-ankle": 16
}

def coco_to_cmu(coco_joints):
    cmu_joints = np.zeros((15,3))
    for cmu_joint, cmu_index in CMU_JOINTS_DEF.items():
        if cmu_joint in COCO_JOINTS_DEF:
            cmu_joints[cmu_index] = coco_joints[COCO_JOINTS_DEF[cmu_joint]]
        elif cmu_joint=='neck':
            cmu_joints[cmu_index] = (coco_joints[COCO_JOINTS_DEF['l-shoulder']] + coco_joints[COCO_JOINTS_DEF['r-shoulder']]) / 2
        elif cmu_joint=='mid-hip':
            cmu_joints[cmu_index] = (coco_joints[COCO_JOINTS_DEF["l-hip"]] + coco_joints[COCO_JOINTS_DEF["r-hip"]]) / 2

    return cmu_joints

class CustomDataset ( Dataset ):
    def __init__(self, dataset_dir, seq_name, cam_list_name, eval_only=False,seq_len=None):
        abs_dataset_dir = osp.abspath ( dataset_dir )
        # exmaple path : 'datasets/panoptic/160224_haggling1/hdImgs/00_00'
        self.eval_only = eval_only
        if not eval_only:
            self.cam_list = CAM_LIST[cam_list_name]
            self.cam_num  = len(self.cam_list )
            self.infos = [ [] for _ in self.cam_list]
            for cam_id, cam_name in enumerate(self.cam_list):
                sub_data_path = f'{abs_dataset_dir}/{seq_name}/hdImgs/{cam_name[0]:02}_{cam_name[1]:02}'
                for frame_id , file_name in enumerate(os.listdir(sub_data_path)):
                    if frame_id % interval == 0:
                        self.infos[cam_id].append(f'{sub_data_path}/{file_name}')
        else:
            self.abs_dataset_dir = abs_dataset_dir
            self.seq_name = seq_name
            self.seq_len = seq_len
            self.gt_list = []
            gt_list_path = f'logs/GT{"-".join([str(l) for l in seq_len])}.npy'
            if os.path.exists(gt_list_path):
                print('[gt_list] laod gt_list from',gt_list_path)
                self.gt_list = np.load(gt_list_path)
            else:
                print('[gt_list] init gt_list to',gt_list_path)
                for seq_id in range(len(seq_name)):
                    for frame_id in tqdm(range(seq_len[seq_id])):
                        frame = interval * frame_id
                        anno_file = f'{abs_dataset_dir}/{seq_name[seq_id]}/hdPose3d_stage1_coco19/body3DScene_{frame:08d}.json'
                        bodies = []
                        all_poses_3d = []
                        all_poses_vis_3d = []

                        if os.path.exists(anno_file):
                            with open(anno_file) as dfile:
                                bodies = json.load(dfile)['bodies']
                        
                        for body in bodies:
                            pose3d = np.array(body['joints19']).reshape((-1, 4))
                            pose3d = pose3d[:15]
                            joints_vis = pose3d[:, -1] > 0.1
                            if not joints_vis[ROOTIDX]:
                                continue

                            all_poses_3d.append(pose3d[:, 0:3])
                            all_poses_vis_3d.append(np.repeat(np.reshape(joints_vis, (-1, 1)), 3, axis=1))
                        self.gt_list.append((all_poses_3d,all_poses_vis_3d))
                np.save(gt_list_path,self.gt_list)
                        
    def __len__(self):
        if not self.eval_only:
            return len ( self.infos[0] )
        else:
            return len(self.anno_files)

    def __getitem__(self, item):
        imgs = list ()
        for cam_id in range(self.cam_num):
            imgs.append ( cv2.imread ( self.infos[cam_id][item] ) )
        return imgs
    
    def get_pose3d(self,item):
        if not self.eval_only:
            
            image_file = self.infos[0][item]
            # jialzhu/data/panoptic/160906_band4/hdPose3d_stage1_coco19/body3DScene_00009961.json
            # /home/tz/repo/mvpose/datasets/panoptic/160422_haggling1/hdPose3d_stage1_coco19/body3DScene_00000000.json
            # jialzhu/data/panoptic/160906_band4/hdImgs/00_03/00_03_00000026.jpg
            folder_path = '/'.join(image_file.split('/')[:-3])
            frame_id = re.search(r'\d{8}', image_file).group()
            anno_file = f'{folder_path}/hdPose3d_stage1_coco19/body3DScene_{frame_id}.json'
            bodies = []
            all_poses_3d = []
            all_poses_vis_3d = []

            if os.path.exists(anno_file):
                with open(anno_file) as dfile:
                    bodies = json.load(dfile)['bodies']
            
            for body in bodies:
                pose3d = np.array(body['joints19']).reshape((-1, 4))
                pose3d = pose3d[:15]
                joints_vis = pose3d[:, -1] > 0.1
                if not joints_vis[ROOTIDX]:
                    continue

                all_poses_3d.append(pose3d[:, 0:3])
                all_poses_vis_3d.append(np.repeat(np.reshape(joints_vis, (-1, 1)), 3, axis=1))
        else:
            all_poses_3d, all_poses_vis_3d = self.gt_list[item]
            
        return all_poses_3d, all_poses_vis_3d
    
    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                tp[i] = 1
                gt_det.append(item["gt_id"])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    def evaluate(self,preds):
        eval_list = []
        total_gt = 0
        M = np.array([[1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]])
        for i,pred in tqdm(enumerate(preds)):
            joints_3d, joints_3d_vis = self.get_pose3d(i)
            if len(joints_3d) > 0:
                joints_3d = [joints.dot(M) * 10 for joints in joints_3d] 
            else:    
                continue
            if preds[i] == False:
                pred = []
                print(f'preds[{i}] is False')
            else:    
                pred = preds[i].copy()
            if len(pred) > 0 :
                pred = [coco_to_cmu(pose.T * 1000) for pose in pred] 
                pass
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(np.sqrt(
                        np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                # score = pose[0, 4]
                score = 1
                eval_list.append({
                    "mpjpe": float(min_mpjpe),
                    "score": float(score),
                    "gt_id": int(total_gt + min_gt)
                })

            total_gt += len(joints_3d)
        
        tb = PrettyTable()
        mpjpe_threshold = np.arange(25, 155, 25)
        aps, recs, mpjpe, recall500 = self._eval_list(eval_list, total_gt, mpjpe_threshold)
        tb.field_names = \
            [f'AP{i}' for i in mpjpe_threshold] + \
            [f'Recall{i}' for i in mpjpe_threshold] + \
            ['Recall500','MPJPE']
        tb.add_row(
            [f'{ap * 100:.2f}' for ap in aps] +
            [f'{re * 100:.2f}' for re in recs] +
            [f'{recall500 * 100:.2f}',f'{mpjpe:.2f}']
        )
        print(tb)
        return aps, recs, recall500, mpjpe

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        eval_list.sort(key=lambda k: k["score"], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item["mpjpe"] < threshold and item["gt_id"] not in gt_det:
                mpjpes.append(item["mpjpe"])
                gt_det.append(item["gt_id"])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf

    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        gt_ids = [e["gt_id"] for e in eval_list if e["mpjpe"] < threshold]
        return len(np.unique(gt_ids)) / total_gt
    
    
    def _eval_list(self, eval_list, total_gt, mpjpe_threshold):
        aps = []
        recs = []
        for t in mpjpe_threshold:
            ap, rec = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            recs.append(rec)
        mpjpe = self._eval_list_to_mpjpe(eval_list)
        recall500 = self._eval_list_to_recall(eval_list, total_gt)
        return aps, recs, mpjpe, recall500
    
    

class BaseDataset ( Dataset ):
    def __init__(self, dataset_dir, range_):
        abs_dataset_dir = osp.abspath ( dataset_dir )
        cam_dirs = [i for i in sorted ( glob ( osp.join ( abs_dataset_dir, '*/' ) ) ) if re.search ( r'\d+/$', i )]
        self.infos = OrderedDict ()
        for cam_idx, cam_dir in enumerate ( cam_dirs ):
            cam_id = int ( re.search ( r'\d+/$', cam_dir ).group ().strip ( '/' ) )

            self.infos[cam_idx] = OrderedDict ()

            img_lists = sorted ( glob ( osp.join ( cam_dir, '*' ) ) )

            for i, img_id in enumerate ( range_ ):
                img_path = img_lists[img_id]
                # img_name = osp.basename ( img_path )
                #
                # pattern = re.compile ( '\d+\.' )
                #
                # img_id = int ( pattern.findall ( img_name )[-1].strip ( '.' ) )  # Not working yet

                self.infos[cam_idx][i] = img_path

    def __len__(self):
        return len ( self.infos[0] )

    def __getitem__(self, item):
        imgs = list ()
        for cam_id in self.infos.keys ():
            # imgs.append ( cv2.imread ( cam_infos[item] ) )
            imgs.append ( cv2.imread ( self.infos[cam_id][item] ) )
        return imgs


class PreprocessedDataset ( Dataset ):
    def __init__(self, dataset_dir):
        self.abs_dataset_dir = osp.abspath ( dataset_dir )
        self.info_files = sorted ( glob ( osp.join ( self.abs_dataset_dir, '*.pickle' ) ),
                                   key=lambda x: int (
                                       osp.basename ( x ).split ( '.' )[0] ) )  # To take %d.infodicts.pickle

    def __len__(self):
        return len ( self.info_files )

    def __getitem__(self, item):
        with open ( self.info_files[item], 'rb' ) as f:
            info_dicts = pickle.load ( f )
        dump_dir = self.abs_dataset_dir
        img_id = int ( osp.basename ( self.info_files[item] ).split ( '.' )[0] )
        for cam_id, info_dict in info_dicts.items ():
            info_dict['image_data'] = np.load ( osp.join ( dump_dir, info_dict['image_path'] ) )

            for pid, person in enumerate ( info_dict[0] ):
                person['heatmap_data'] = np.load ( osp.join ( dump_dir, person['heatmap_path'] ) )
                person['cropped_img'] = np.load ( osp.join ( dump_dir, person['cropped_path'] ) )
        return info_dicts
