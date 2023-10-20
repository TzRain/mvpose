
from torch.utils.data import Dataset
import numpy as np
import re
from glob import glob
import pickle
import os
import os.path as osp
from collections import OrderedDict
import cv2

JOINTS_DEF = {
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

class CustomDataset ( Dataset ):
    def __init__(self, dataset_dir, seq_name, cam_list_name):
        abs_dataset_dir = osp.abspath ( dataset_dir )
        # exmaple path : 'datasets/panoptic/160224_haggling1/hdImgs/00_00'
        self.cam_list = CAM_LIST[cam_list_name]
        self.cam_num  = len(self.cam_list )
        self.infos = [ [] for _ in self.cam_list]
        for cam_id, cam_name in enumerate(self.cam_list):
            sub_data_path = f'{abs_dataset_dir}/{seq_name}/hdImgs/{cam_name[0]:02}_{cam_name[1]:02}'
            for frame_id , file_name in enumerate(os.listdir(sub_data_path)):
                if frame_id % interval == 0:
                    self.infos[cam_id].append(f'{sub_data_path}/{file_name}')
                        
    def __len__(self):
        return len ( self.infos[0] )

    def __getitem__(self, item):
        imgs = list ()
        for cam_id in range(self.cam_num):
            imgs.append ( cv2.imread ( self.infos[cam_id][item] ) )
        return imgs
    
    def get_pose3d(self,item):
        image_file = self.infos[0][item]
        # jialzhu/data/panoptic/160906_band4/hdPose3d_stage1_coco19/body3DScene_00009961.json
        # jialzhu/data/panoptic/160906_band4/hdImgs/00_03/00_03_00000026.jpg
        folder_path = '/'.join(image_file.split('/')[:-2])
        frame_id = re.search(r'\d{8}', image_file).group()
        anno_file = f'{folder_path}/hdPose3d_stage1_coco19/body3DScene_{frame_id}.json'
        
        bodies = []
        all_poses_3d = []
        all_poses_vis_3d = []

        if os.path.exists(file_path):
            with open(anno_file) as dfile:
                bodies = json.load(dfile)['bodies']
        
        for body in bodies:
            pose3d = np.array(body['joints19']).reshape((-1, 4))
            pose3d = pose3d[:len(JOINTS_DEF)]
            joints_vis = pose3d[:, -1] > 0.1
            if not joints_vis[ROOTIDX]:
                continue

            all_poses_3d.append(pose3d[:, 0:3])
            all_poses_vis_3d.append(np.repeat(np.reshape(joints_vis, (-1, 1)), 3, axis=1))
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
        for i,pred in enumerate(preds)
            joints_3d, joints_3d_vis = self.get_pose3d(i)
            if len(joints_3d) == 0:
                continue

            pred = preds[i].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    vis = gt_vis[:, 0] > 0
                    mpjpe = np.mean(np.sqrt(
                        np.sum((pose[vis, 0:3] - gt[vis]) ** 2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
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
        logger.info(tb)

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
    
    @staticmethod
    def _eval_list(eval_list, total_gt, mpjpe_threshold):
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
