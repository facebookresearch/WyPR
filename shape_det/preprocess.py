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

data_path = '.'

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

def compute_adj_mat():
    """ pre-compute adj mat for GSS """    
    for split in ['train', 'val']:
        files = [line.rstrip() for line in open('../wypr/dataset/scannet/meta_data/scannetv2_%s.txt'%split)]
        for scene_id in tqdm(files):
            out_fname = os.path.join(data_path, 'cgal_output', scene_id+'.pkl')
            if os.path.exists(out_fname):
                continue
            cgal_txt_f = os.path.join(data_path, 'cgal_output', scene_id+'.txt')
            with open(cgal_txt_f, 'r') as fin:
                cgal_txt = fin.readlines()
            # 2nd last row is empty line; last row are the missed points
            n_region = len(cgal_txt) - 2
            shapes = []
            points = np.load(os.path.join('../wypr/dataset/scannet/scannet_all_points', scene_id+'_vert.npy'))
            F0 = np.ones(points.shape[0]) * n_region
            for i in range(len(cgal_txt[:-2])):
                row = cgal_txt[i]
                _points = row.rstrip(' \n').split(' ')
                row_list = [int(item) for item in _points]
                pcd_i = o3d.geometry.PointCloud()
                idx_i = np.array(row_list)
                pcd_i.points = o3d.utility.Vector3dVector(points[:, :3][idx_i])
                shapes += [pcd_i]
                F0[idx_i] = i
            adj_mat, A0 = _calc_adjacency_matrix(shapes, n_region)
            return_dict = {'adj_mat': adj_mat, 'A0':A0}
            with open(out_fname, 'wb') as f:
                pickle.dump(return_dict, f)

if __name__=='__main__':
    compute_adj_mat()