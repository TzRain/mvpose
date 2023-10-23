import os
import sys
import numpy as np
import argparse
project_root = os.path.abspath ( os.path.join ( os.path.dirname ( __file__ ), '..', '..' ) )
if __name__ == '__main__':
    if project_root not in sys.path:
        sys.path.append ( project_root )
from src.m_utils.base_dataset import  CustomDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser ()
    parser.add_argument('--seq', nargs='+', default=None )
    parser.add_argument('--pose', nargs='+', default=None )
    args = parser.parse_args ()
    print('[CustomDataset] init_data_set')
    eval_dataset = CustomDataset('datasets/panoptic', args.seq, None, eval_only=True)
    poses3ds = []
    pose_dir = args.pose
    if not isinstance(pose_dir,list):
        pose_dir = [pose_dir]
    poses3ds = []
    print('[poses3ds] load poses')
    for pose in pose_dir:
        poses3ds.append(np.load(f'{pose}/poses3ds.npy'))
    all_pose3d = np.concatenate(poses3ds, axis=0)
    eval_dataset.evaluate(all_pose3d)