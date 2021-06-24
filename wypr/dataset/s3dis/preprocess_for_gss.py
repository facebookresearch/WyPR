# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import open3d as o3d
import numpy as np
import argparse
import pickle

from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

data_path ='s3dis_processed'

parser = argparse.ArgumentParser('prepare the s3dis dataset')
parser.add_argument('--n_proc', '-n', type=int, default=64, help='Number of processes to use.')
args = parser.parse_args()

def _calc_adjacency_matrix(shapes, n_region, TAU=0.2):
    assert len(shapes) == n_region
    Adjacency = np.zeros((n_region, n_region))
    for i in tqdm(range(n_region)):
        Adjacency[i, i] = 1
        try:
            shape_i = copy.deepcopy(shapes[i])
            # perturb slightly    
            shape_i.points = o3d.utility.Vector3dVector(np.asarray(shape_i.points) * (1 - TAU/2 + np.random.rand(len(shape_i.points), 3)*TAU) )
            hull_i = shape_i.compute_convex_hull()
        except RuntimeError:
            continue

        for j in range(i+1, n_region):
            try:
                shape_j = copy.deepcopy(shapes[j])
                shape_j.points = o3d.utility.Vector3dVector(np.asarray(shape_j.points) * (1 - TAU/2 + np.random.rand(len(shape_j.points), 3)*TAU) )
                hull_j = shape_j.compute_convex_hull()
                if hull_j[0].is_intersecting(hull_i[0]):
                    Adjacency[i, j] = Adjacency[j, i] = 1   
            except RuntimeError:
                continue
    dic = {i : {i} ^ set(np.flatnonzero(Adjacency[i])) for i in range(n_region)}
    return Adjacency, dic

def _compute_adj_mat(scene_id):
    out_fname = os.path.join(data_path, scene_id+'.pkl')
    if os.path.exists(out_fname):
        return 
    print('computing', out_fname)
  
    F0 = np.load(os.path.join(data_path, scene_id+'_shape.npy'))
    points = np.load(os.path.join(data_path, scene_id+'.npy'))
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
    scan_names = [line.rstrip('/Annotations\n') for line in open('meta/anno_paths.txt')]
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(_compute_adj_mat), scan_names)
    else:
        for _path in scan_names:
            _compute_adj_mat(_path)


if __name__=='__main__':
    compute_adj_mat()