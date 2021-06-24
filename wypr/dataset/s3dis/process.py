# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import open3d as o3d
import glob
import numpy as np
from tqdm import tqdm
from scipy import spatial
import argparse
from multiprocessing import Pool
from functools import partial

import sys
# https://stackoverflow.com/questions/6809402/python-maximum-recursion-depth-exceeded-while-calling-a-python-object
sys.setrecursionlimit(10000)

anno_path = 'meta/anno_paths.txt'
DATA_PATH = 'Stanford3dDataset_v1.2_Aligned_Version'
PCD_PATH = 's3dis_processed'
CGAL_PATH = '../../../shape_det/build/region_growing_on_point_set_3'

sem_map = {'ceiling':0, 'floor':1, 'wall':2, 'column':3,'beam':4, 'window':5, 'door':6,
           'table':7, 'chair':8, 'bookcase':9, 'sofa':10, 'board':11, 'clutter':12, 'stairs':13}

parser = argparse.ArgumentParser('prepare the s3dis dataset')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')
args = parser.parse_args()

def _convert_pc2npy(_path):
    print(_path)
    elements = str(_path).split('/')
    out_file_name = os.path.join(PCD_PATH, elements[-3], elements[-2]+'_annot.npy')
    if os.path.exists(out_file_name):
        return

    data_list = []
    inst_id = 0
    for f in glob.glob(os.path.join(_path, '*.txt')):
        class_name = os.path.basename(f).split('_')[0]
        pc = np.loadtxt(f)
        labels = np.ones((pc.shape[0], 1)) * sem_map[class_name]
        inst_labels = np.zeros((pc.shape[0], 1)) + inst_id
        # Nx8 XYZ RGB sem_L inst_L
        data_list.append(np.concatenate([pc, labels, inst_labels], 1))  
        inst_id += 1

    pc_label = np.concatenate(data_list, 0)
    os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
    np.save(out_file_name, pc_label)

def convert_pc2npy():
    """
    Convert original dataset files to npy file (each line is XYZRGBL).
    We aggregated all the points from each instance in the room.
    """
    anno_paths = [line.rstrip() for line in open('meta/anno_paths.txt')]
    anno_paths = [os.path.join(DATA_PATH, p) for p in anno_paths]
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(_convert_pc2npy), anno_paths)
    else:
        for _path in anno_paths:
            _convert_pc2npy(_path)

convert_pc2npy()

def _convert2xyzn(line):
    print(line)
    area, room, _ = line.split('/')
    if os.path.exists(os.path.join(PCD_PATH, area, room+'.xyzn')):
        return
    
    file_path = os.path.join(PCD_PATH, area, room+'_annot.npy')
    assert os.path.exists(file_path)
    points = np.load(file_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255)
    pcd.estimate_normals()
    o3d.io.write_point_cloud(os.path.join(PCD_PATH, area, room+'.xyzn'), pcd)

def convert2xyzn():
    """ convert the *.txt files to *.xyzn so that cgal can take as input """
    print("==> convert the *.txt files to *.xyzn")
    anno_paths = [line.rstrip() for line in open('meta/anno_paths.txt')]
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(_convert2xyzn), anno_paths)
    else:
        for _path in anno_paths:
            _convert2xyzn(_path)

convert2xyzn()

def _run_shape_det(line):
    abs_out_path = os.path.abspath(PCD_PATH)
    area, room, _ = line.split('/')
    file_path = os.path.join(abs_out_path, area, room+'.xyzn')
    assert os.path.exists(file_path)

    output_name = os.path.join(abs_out_path, area, room+'.ply')
    output_txt = os.path.join(abs_out_path, area, room+'.txt')
    if not os.path.exists(output_name) or not os.path.exists(output_txt):
        os.system(CGAL_PATH +'\t'+ file_path +'\t'+ output_name +'\t'+  output_txt  +'\n')

def run_shape_det():
    """ generate bash scripts
        use absolute path so that these scripts could be used in cgal directory 
    """
    os.makedirs('cgal_input', exist_ok=True)
    anno_paths = [line.rstrip() for line in open('meta/anno_paths.txt')]
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(_run_shape_det), anno_paths)
    else:
        for _path in anno_paths:
            _run_shape_det(_path)

run_shape_det()


def _fix_unassigned_points(line):
    print(line)
    abs_out_path = os.path.abspath(PCD_PATH)
    area, room, _ = line.split('/')
    output_f = os.path.join(abs_out_path, area, room+'_shape.npy')
    if os.path.exists(output_f):
        return
    cgal_txt_f = os.path.join(abs_out_path, area, room+'.txt')
    assert os.path.exists(cgal_txt_f)
    with open(cgal_txt_f, 'r') as fin:
        cgal_txt = fin.readlines()
    pcd_path = os.path.join(abs_out_path, area, room+'.xyzn')
    assert os.path.exists(pcd_path)
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    F0 = np.ones(points.shape[0]) * -1
    for i in range(len(cgal_txt[:-2])):
        try:
            row_list = [int(item) for item in cgal_txt[i].rstrip(' \n').split(' ')]
        except ValueError:
            print(i, cgal_txt_f)
            import pdb; pdb.set_trace()
        F0[np.array(row_list)] = i
    
    assigned_idx = np.where(F0!=-1)[0]
    unassigned_idx = np.where(F0==-1)[0]
    assigned_points = points[assigned_idx, :3]
    unassigned_points = points[unassigned_idx, :3]
    dist, idx = spatial.KDTree(assigned_points).query(unassigned_points)
    F0[unassigned_idx] = F0[assigned_idx][idx]
    assert np.sum(F0==-1) == 0
    np.save(output_f, F0)

def fix_unassigned_points():
    """ assign the un-assinged label to its nearest assinged points """
    anno_paths = [line.rstrip() for line in open('meta/anno_paths.txt')]
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(_fix_unassigned_points), anno_paths)
    else:
        for _path in anno_paths:
            _fix_unassigned_points(_path)

fix_unassigned_points()