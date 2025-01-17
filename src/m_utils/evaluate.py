
import os
import sys
import os.path as osp
import json
import pickle
from datetime import datetime
current_datetime = datetime.now()
time_stamp = current_datetime.strftime("%y-%m-%dT%H:%M")

project_root = os.path.abspath ( os.path.join ( os.path.dirname ( __file__ ), '..', '..' ) )
if __name__ == '__main__':
    if project_root not in sys.path:
        sys.path.append ( project_root )
import coloredlogs, logging

logger = logging.getLogger ( __name__ )
coloredlogs.install ( level='DEBUG', logger=logger )

from src.models.model_config import model_cfg
import time
import scipy.io as scio
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from prettytable import PrettyTable
import csv
from copy import deepcopy
import torch
from glob import glob
import cv2
from torch.utils.data import DataLoader
from src.m_utils.base_dataset import BaseDataset, PreprocessedDataset, CustomDataset
from src.models.estimate3d import MultiEstimator
from src.m_utils.transformation import coco2shelf3D
from src.m_utils.numeric import vectorize_distance
from src.m_utils.mem_dataset import MemDataset

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


def is_right(model_start_point, model_end_point, gt_strat_point, gt_end_point, alpha=0.5):
    bone_lenth = np.linalg.norm ( gt_end_point - gt_strat_point )
    start_difference = np.linalg.norm ( gt_strat_point - model_start_point )
    end_difference = np.linalg.norm ( gt_end_point - model_end_point )
    return ((start_difference + end_difference) / 2) <= alpha * bone_lenth


def numpify(info_dicts):
    for info_dict in info_dicts.values ():
        info_dict['image_data'] = info_dict['image_data'].squeeze ().numpy ()
        for person in info_dict[0]:
            person['heatmap_data'] = person['heatmap_data'].squeeze ().numpy ()
            person['cropped_img'] = person['cropped_img'].squeeze ().numpy ()
    return info_dicts


def evaluate(model, actor3D, range_, loader, is_info_dicts=False, dump_dir=None, save_dir=None, start_from=0):
    poses3ds = []
    for idx, imgs in enumerate ( tqdm ( loader ) ):
        if idx < start_from:
            continue
        img_id = range_[idx] if range_ is not None else None
        try:
            if is_info_dicts:
                info_dicts = numpify ( imgs )
                model.dataset = MemDataset ( info_dict=info_dicts, camera_parameter=camera_parameter,
                                             template_name='Unified' )
                poses3d = model._estimate3d ( 0, show=False )
            else:
                this_imgs = list ()
                for img_batch in imgs:
                    this_imgs.append ( img_batch.squeeze ().numpy () )
                poses3d = model.predict ( imgs=this_imgs, camera_parameter=camera_parameter, template_name='Unified',
                                          show=False )
        except Exception as e:
            logger.critical ( e )
            poses3d = False
        poses3ds.append(poses3d)
        if save_dir is not None:
            np.save(f'{save_dir}/poses_{idx}.npy',poses3d)
        
    return poses3ds
    check_result = np.zeros ( (len ( actor3D[0] ), len ( actor3D ), 10), dtype=np.int32 )
    accuracy_cnt = 0
    error_cnt = 0
    #     for pid in range ( len ( actor3D ) ):
    #         if actor3D[pid][img_id][0].shape == (1, 0) or actor3D[pid][img_id][0].shape == (0, 0):

    #             continue

    #         if not poses3d:
    #             check_result[img_id, pid, :] = -1
    #             logger.error ( f'Cannot get any pose in img:{img_id}' )
    #             continue
    #         model_poses = np.stack ( [coco2shelf3D ( i ) for i in deepcopy ( poses3d )] )
    #         gt_pose = actor3D[pid][img_id][0]
    #         dist = vectorize_distance ( np.expand_dims ( gt_pose, 0 ), model_poses )
    #         model_pose = model_poses[np.argmin ( dist[0] )]

    #         bones = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [12, 13]]
    #         for i, bone in enumerate ( bones ):
    #             start_point, end_point = bone
    #             if is_right ( model_pose[start_point], model_pose[end_point], gt_pose[start_point],
    #                           gt_pose[end_point] ):
    #                 check_result[img_id, pid, i] = 1
    #                 accuracy_cnt += 1
    #             else:
    #                 check_result[img_id, pid, i] = -1
    #                 error_cnt += 1
    #         gt_hip = (gt_pose[2] + gt_pose[3]) / 2
    #         model_hip = (model_pose[2] + model_pose[3]) / 2
    #         if is_right ( model_hip, model_pose[12], gt_hip, gt_pose[12] ):
    #             check_result[img_id, pid, -1] = 1
    #             accuracy_cnt += 1
    #         else:
    #             check_result[img_id, pid, -1] = -1
    #             error_cnt += 1
    # bone_group = OrderedDict (
    #     [('Head', np.array ( [8] )), ('Torso', np.array ( [9] )), ('Upper arms', np.array ( [5, 6] )),
    #      ('Lower arms', np.array ( [4, 7] )), ('Upper legs', np.array ( [1, 2] )),
    #      ('Lower legs', np.array ( [0, 3] ))] )

    # total_avg = np.sum ( check_result > 0 ) / np.sum ( np.abs ( check_result ) )
    # person_wise_avg = np.sum ( check_result > 0, axis=(0, 2) ) / np.sum ( np.abs ( check_result ), axis=(0, 2) )

    # bone_wise_result = OrderedDict ()
    # bone_person_wise_result = OrderedDict ()
    # for k, v in bone_group.items ():
    #     bone_wise_result[k] = np.sum ( check_result[:, :, v] > 0 ) / np.sum ( np.abs ( check_result[:, :, v] ) )
    #     bone_person_wise_result[k] = np.sum ( check_result[:, :, v] > 0, axis=(0, 2) ) / np.sum (
    #         np.abs ( check_result[:, :, v] ), axis=(0, 2) )

    # tb = PrettyTable ()
    # tb.field_names = ['Bone Group'] + [f'Actor {i}' for i in range ( bone_person_wise_result['Head'].shape[0] )] + [
    #     'Average']
    # list_tb = [tb.field_names]
    # for k, v in bone_person_wise_result.items ():

    #     this_row = [k] + [np.char.mod ( '%.4f', i ) for i in v] + [np.char.mod ( '%.4f', np.sum ( v ) / len ( v ) )]
    #     list_tb.append ( [float ( i ) if isinstance ( i, type ( np.array ( [] ) ) ) else i for i in this_row] )
    #     tb.add_row ( this_row )
    # this_row = ['Total'] + [np.char.mod ( '%.4f', i ) for i in person_wise_avg] + [
    #     np.char.mod ( '%.4f', np.sum ( person_wise_avg ) / len ( person_wise_avg ) )]
    # tb.add_row ( this_row )
    # list_tb.append ( [float ( i ) if isinstance ( i, type ( np.array ( [] ) ) ) else i for i in this_row] )
    # if dump_dir:
    #     np.save ( osp.join ( dump_dir, time.strftime ( str ( model_cfg.testing_on ) + "_%Y_%m_%d_%H_%M",
    #                                                    time.localtime ( time.time () ) ) ), check_result )
    #     with open ( osp.join ( dump_dir,
    #                            time.strftime ( str ( model_cfg.testing_on ) + "_%Y_%m_%d_%H_%M.csv",
    #                                            time.localtime ( time.time () ) ) ), 'w' ) as f:
    #         writer = csv.writer ( f )
    #         writer.writerows ( list_tb )
    #         writer.writerow ( [model_cfg] )
    # print ( tb )
    # print ( model_cfg )
    # return check_result, list_tb


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument ( '-d', nargs='+', dest='datasets', required=True, choices=['Shelf', 'Campus','panoptic'] )
    parser.add_argument ( '-dumped', nargs='+', dest='dumped_dir', default=None )
    parser.add_argument('--seq', type=str, help='A string argument')
    parser.add_argument('--cam', type=str, help='A string argument')
    parser.add_argument('--load-pose', type=str, help='pose dir', default=None)
    args = parser.parse_args ()
    seq, cam = args.seq, args.cam
    test_model = MultiEstimator ( cfg=model_cfg ) if args.load_pose is None else None

    for dataset_idx, dataset_name in enumerate ( args.datasets ):
        model_cfg.testing_on = dataset_name

        if dataset_name == 'Shelf':
            dataset_path = model_cfg.shelf_path
            test_range = range ( 300, 600 )
            gt_path = dataset_path

        elif dataset_name == 'Campus':
            dataset_path = model_cfg.campus_path
            test_range = [i for i in range ( 350, 471 )] + [i for i in range ( 650, 751 )]
            gt_path = dataset_path
            
        elif dataset_name == 'panoptic':
            dataset_path = 'datasets/panoptic'
            test_range = None

        else:
            dataset_path = model_cfg.panoptic_ultimatum_path
            test_range = range ( 4500, 4900 )
            gt_path = osp.join ( dataset_path, '..' )
        
        if dataset_name != 'panoptic':

            with open ( osp.join ( dataset_path, 'camera_parameter.pickle' ),
                        'rb' ) as f:
                camera_parameter = pickle.load ( f )
        else:
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

            M = np.array([[1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0]])
            
            # stage 0 
            K = np.stack([np.array(camera['K']) for camera in selected_cameras], axis=0).astype(np.float32)
            T = np.stack([np.array(camera['t']) for camera in selected_cameras], axis=0).astype(np.float32)
            R = np.stack([np.array(camera['R']) for camera in selected_cameras], axis=0).astype(np.float32)
            # stage 1
            T = np.stack([-np.dot(R[i].T, T[i]) * 10.0 / 1000 for i in range(len(selected_cameras))], axis=0).astype(np.float64)
            T = np.stack([np.dot(M.T,T[i]) for i in range(len(selected_cameras))], axis=0).astype(np.float64)
            # stage 2
            R = np.stack([R[i].dot(M) for i in range(len(selected_cameras))], axis=0).astype(np.float64)
            T = np.stack([ -np.dot(R[i], T[i]) for i in range(len(selected_cameras))], axis=0).astype(np.float64)
            RT = np.stack([ np.concatenate((R[i], T[i]),axis=-1) for i in range(len(selected_cameras))], axis=0).astype(np.float64)
            P = np.stack([np.dot(K[i], RT[i]) for i in range(len(selected_cameras))], axis=0).astype(np.float64)

            camera_parameter = {
                'K': K,
                'RT': RT,
                'P': P,
            }
            
        if args.dumped_dir:
            test_dataset = PreprocessedDataset ( args.dumped_dir[dataset_idx])
            logger.info ( f"Using pre-processed datasets {args.dumped_dir[dataset_idx]} for quicker evaluation" )
            test_loader = DataLoader ( test_dataset, batch_size=1, pin_memory=True, num_workers=6, shuffle=False )
        else:
            test_dataset = BaseDataset ( dataset_path, test_range ) if dataset_name != 'panoptic' else CustomDataset ( dataset_path, seq, cam )
            test_loader = DataLoader ( test_dataset, batch_size=1, pin_memory=True, num_workers=12, shuffle=False )

        actorsGT = scio.loadmat ( osp.join ( gt_path, 'actorsGT.mat' ) ) if dataset_name != 'panoptic' else None
        test_actor3D = actorsGT['actor3D'][0] if dataset_name != 'panoptic' else None
        if dataset_name == 'panoptic':
            save_dir = f'logs/{dataset_name}_{seq}_{cam}_{time_stamp}'
            os.makedirs(save_dir)
        if args.load_pose is not None:
            poses3ds = np.load(f'{args.load_pose}/poses3ds.npy')
            poses3ds = [pose if pose!=False else [] for pose in poses3ds]
            aps, recs, recall500, mpjpe = test_dataset.evaluate(poses3ds)
        else:
            poses3ds = evaluate ( test_model, test_actor3D, test_range, test_loader, is_info_dicts=bool ( args.dumped_dir ),
                   dump_dir=osp.join ( project_root, 'result' ),save_dir=save_dir )
        if dataset_name == 'panoptic':
            np.save (  f'{save_dir}/poses3ds.npy', poses3ds )
            
