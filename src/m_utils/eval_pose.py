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
    
    
    poses3ds = []
    pose_dir = args.pose
    if not isinstance(pose_dir,list):
        pose_dir = [pose_dir]
    poses3ds = []
    seq_len = []
    print('[poses3ds] load poses')
    for i,pose in enumerate(pose_dir):
        pose3ds = np.load(f'{pose}/poses3ds.npy')
        poses3ds.append(pose3ds)
        seq_len.append(pose3ds.shape[0])
    all_pose3d = np.concatenate(poses3ds, axis=0)
    print('[seq_len]',seq_len)
    print('[seq_name]',args.seq)
    print('[CustomDataset] init_data_set')
    eval_dataset = CustomDataset('datasets/panoptic', args.seq, None, eval_only=True, seq_len=seq_len)
    eval_dataset.evaluate(all_pose3d)