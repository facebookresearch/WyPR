# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import open3d as o3d
import glob
import numpy as np
import copy
from tqdm import tqdm

data_path = '/data/fb_data'

def generate_pcd_from_mesh():
    for split in ['train', 'val']:
        files = [line.rstrip() for line in open('../wypr/dataset/scannet/meta_data/scannetv2_%s.txt'%split)]
        out_path = os.path.join(data_path, 'processed', split)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        for scan_name in tqdm(files):
            mesh_file = os.path.join('../wypr/dataset/scannet/scans', scan_name, scan_name + '_vh_clean_2.ply')
            out_file = os.path.join(out_path, scan_name + '.ply')
            if not os.path.exists(out_file):
                pcd_old = o3d.io.read_point_cloud(mesh_file)
                pcd = o3d.geometry.PointCloud()
                pcd.points = pcd_old.points
                pcd.colors = pcd_old.colors
                o3d.io.write_point_cloud(out_file, pcd)

def generate_input():
    for split in ['train', 'test']:
        data_folder = os.path.join(data_path, 'processed', split)
        out_path = os.path.join(data_path, 'cgal_input', split)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        files = glob.glob(data_folder + '/*.ply')
        for f in files:
            scene_name = f.replace(".ply", ".xyzn").split("/")[-1]
            pcd = o3d.io.read_point_cloud(f)
            print(scene_name, pcd)
            pcd.estimate_normals()
            assert o3d.io.write_point_cloud(os.path.join(out_path, scene_name), pcd)

def generate_script():
    for split in ['train', 'test']:
        data_folder = os.path.join(data_path, 'cgal_input', split)
        files = glob.glob(data_folder + '/*.xyzn')
        for i, f in enumerate(files):
            with open(os.path.join(data_path, 'cgal_input', split + '%d.sh'%(i/100)), 'a') as fout:
                output_name = os.path.join(data_path, 'cgal_output/', f.replace(".xyzn", ".ply").split("/")[-1])
                output_txt = os.path.join(data_path, 'cgal_output/', f.replace(".xyzn", ".txt").split("/")[-1])
                fout.write('./region_growing_on_point_set_3' + '\t' +
                           os.path.join(data_path, f) + '\t' +
                           os.path.join(data_path, output_name) + '\t' + 
                           os.path.join(data_path, output_txt) + '\n')

if __name__=='__main__':
    generate_pcd_from_mesh()
    generate_input()
    generate_script()