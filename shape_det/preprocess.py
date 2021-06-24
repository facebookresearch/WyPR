# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import open3d as o3d
import numpy as np
import copy
import pickle
import argparse
from multiprocessing import Pool
from functools import partial
from scipy import spatial

data_path = '/data/fb_data'

parser = argparse.ArgumentParser('prepare the s3dis dataset')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')
args = parser.parse_args()


def _fix_unassigned_points(line):
    print(line)
    scene_id = line.split('/')[-1].rstrip('.txt')
    output_f = os.path.join(data_path, 'cgal_output', scene_id+'_shape.npy')
    if os.path.exists(output_f):
        return
    assert os.path.exists(line)
    with open(line, 'r') as fin:
        cgal_txt = fin.readlines()
    pcd_path = os.path.join(data_path, 'cgal_input/all', scene_id+'.xyzn')
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
    anno_paths = glob.glob(os.path.join(data_path, 'cgal_output_0929', '*.txt'))

    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(_fix_unassigned_points), anno_paths)
    else:
        for _path in anno_paths:
            _fix_unassigned_points(_path)

def _calc_adjacency_matrix(shapes, n_region, TAU=0.2):
    assert len(shapes) == n_region
    Adjacency = np.zeros((n_region, n_region))
    for i in range(n_region):
        Adjacency[i, i] = 1
        for j in range(i+1, n_region):
            shape_i = copy.deepcopy(shapes[i])
            # perturb slightly    
            shape_i.points = o3d.utility.Vector3dVector(np.asarray(shape_i.points) * (1 - TAU/2 + np.random.rand(len(shape_i.points), 3)*TAU) )
            hull_i = shape_i.compute_convex_hull()

            shape_j = copy.deepcopy(shapes[j])
            shape_j.points = o3d.utility.Vector3dVector(np.asarray(shape_j.points) * (1 - TAU/2 + np.random.rand(len(shape_j.points), 3)*TAU) )
            hull_j = shape_j.compute_convex_hull()

            if hull_j[0].is_intersecting(hull_i[0]):
                Adjacency[i, j] = Adjacency[j, i] = 1   
    dic = {i : {i} ^ set(np.flatnonzero(Adjacency[i])) for i in range(n_region)}
    return Adjacency, dic

def _compute_adj_mat(scene_id):
    """ pre-compute adj mat for GSS """    
    out_fname = os.path.join(data_path, 'cgal_output', scene_id+'.pkl')
    if os.path.exists(out_fname):
        return
    print(out_fname)
    F0 = np.load(os.path.join(data_path, 'cgal_output', scene_id+'_shape.npy'))
    points = np.load(os.path.join('../wypr/dataset/scannet/scannet_all_points', scene_id+'_vert.npy'))
    n_region = len(np.unique(F0))
    shapes = []
    for i in range(n_region):
        pcd_i = o3d.geometry.PointCloud()
        idx_i = np.where(F0 == i)[0]
        pcd_i.points = o3d.utility.Vector3dVector(points[idx_i, 0:3])
        shapes += [pcd_i]

    adj_mat, A0 = _calc_adjacency_matrix(shapes, n_region)
    return_dict = {'adj_mat': adj_mat, 'A0':A0}
    with open(out_fname, 'wb') as f:
        pickle.dump(return_dict, f)

def compute_adj_mat():
    """ pre-compute adj mat for GSS """    
    scan_names = [line.rstrip() for line in open('../wypr/dataset/scannet/meta_data/scannetv2_train.txt')]
    scan_names += [line.rstrip() for line in open('../wypr/dataset/scannet/meta_data/scannetv2_val.txt')]

    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(_compute_adj_mat), scan_names)
    else:
        for _path in scan_names:
            _compute_adj_mat(_path)

if __name__=='__main__':
    fix_unassigned_points()
    compute_adj_mat()