
from torch.utils.data import Dataset
import numpy as np
import re
from glob import glob
import pickle
import os
import os.path as osp
from collections import OrderedDict
import cv2

CAM_LIST={
    'CMU0_ori': [(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)],  # Origin order in MvP
    'CMU0' : [(0, 3), (0, 6),(0, 12),(0, 13), (0, 23)],
    'CMU0ex' : [(0, 3), (0, 6), (0, 12),(0, 13), (0, 23), (0, 10), (0, 16)],
    'CMU1' : [(0, 1),(0, 2),(0, 3),(0, 4),(0, 6),(0, 7),(0, 10)],  
    'CMU2' : [(0, 12), (0, 16), (0, 18), (0, 19), (0, 22), (0, 23), (0, 30)],
    'CMU3': [(0, 10), (0, 12), (0, 16), (0, 18)],
    'CMU4' : [(0, 6), (0, 7), (0, 10), (0, 12), (0, 16), (0, 18), (0, 19), (0, 22), (0, 23), (0, 30)],
}
# cam_list = list(set(CAM_LIST['CMU0ex'] + CAM_LIST['CMU1'] + CAM_LIST['CMU2'] + CAM_LIST['CMU3'] + CAM_LIST['CMU4']))
seq_list = ['160906_pizza1', '160422_haggling1', '160906_ian5', '160906_band4'],
interval = 12

class CustomDataset ( Dataset ):
    def __init__(self, dataset_dir, cam_list):
        abs_dataset_dir = osp.abspath ( dataset_dir )
        # exmaple path : 'datasets/panoptic/160224_haggling1/hdImgs/00_00'
        self.cam_list = cam_list
        self.cam_num  = len(cam_list)
        for seq_name in seq_list:
           for cam_id, cam_name in enumerate(self.cam_list):
                sub_data_path = f'{abs_dataset_dir}/{seq_name}/hdImgs/{cam_name[0]:02}_{cam_name[1]:02}'
                for frame_id , file_name in enumerate(os.listdir(sub_data_path)):
                    if frame_id % interval == 0:
                        self.infos[cam_id].append(f'{sub_data_path}/{file_name}')
                        
    def __len__(self):
        return len ( self.infos[0] )

    def __getitem__(self, item):
        imgs = list ()
        for cam_id in self.cam_num:
            imgs.append ( cv2.imread ( self.infos[cam_id][item] ) )
        return imgs

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
