# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import datetime
import numpy as np
import open3d as o3d
from multiprocessing import Pool
from functools import partial

sem_map = {'ceiling':0, 'floor':1, 'wall':2, 'column':3,'beam':4, 'window':5, 'door':6,
           'table':7, 'chair':8, 'bookcase':9, 'sofa':10, 'board':11, 'clutter':12, 'stairs':13}

parser = argparse.ArgumentParser('prepare the s3dis dataset')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')
parser.add_argument('--output_dir', type=str, default='s3dis_all_points',
                    help='where to save')
parser.add_argument('--s3dis_dir', type=str, default='s3dis_processed',
                    help='data path')


def export(pcd_file, cgal_file, compute_normal=False):
    """ points are XYZ RGB (RGB in 0-255),
        box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """ 
    pcd = np.load(pcd_file)
    shape_labels = np.load(cgal_file)
    assert len(shape_labels) == len(pcd)
    
    all_labels = np.unique(pcd[:, -2])
    num_inst = int(np.max(pcd[:, -1]))

    room_center = np.zeros((pcd.shape[0], 3))
    room_size = np.zeros((pcd.shape[0], 3))

    bboxes = np.zeros((num_inst, 7))
    for i in range(num_inst):
        ins_inds = pcd[:, -1] == i
        ins_pt = pcd[ins_inds, :3]
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        xmin, ymin, zmin = ins_pt.min(0)
        xmax, ymax, zmax = ins_pt.max(0)
        label_id = int(pcd[ins_inds, -2][0])
        bboxes[i] = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2,
            xmax-xmin, ymax-ymin, zmax-zmin, label_id])

    # compute surface normal
    if not compute_normal:
        return [pcd, shape_labels, bboxes, None]
    else:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd[:, :3])
        point_cloud.colors = o3d.utility.Vector3dVector(pcd[:, 3:6] / 255)
        point_cloud.estimate_normals()
        normal = np.asarray(point_cloud.normals)
        return [pcd, shape_labels, bboxes, normal]

def exported(prefix):
    return os.path.isfile(prefix+'/vert.npy') and os.path.isfile(prefix+'/sem_label.npy') \
        and os.path.isfile(prefix+'/bbox.npy') and os.path.isfile(prefix+'/shape.npy') \
        and os.path.isfile(prefix+'/normal.npy')

def export_scan(scan_name, args):        
    print(scan_name, datetime.datetime.now())
    output_prefix = os.path.join(args.output_dir, scan_name) 
    os.makedirs(output_prefix, exist_ok=True)
    if not exported(output_prefix):
        area = scan_name.split('/')[-2]
        room = scan_name.split('/')[-1]
        pcd_file = os.path.join(args.s3dis_dir, area, room + '_annot.npy')
        assert os.path.exists(pcd_file), pcd_file
        
        cgal_file = os.path.join(args.s3dis_dir, area, room + '_shape.npy')
        assert os.path.exists(cgal_file), cgal_file

        pcd, shape_labels, bboxes, normal = export(pcd_file, cgal_file, True)

        num_shapes = len(np.unique(shape_labels)) - 1
        print('Num of primitives: ', num_shapes)
        print('Num of care instances/boxes: ', bboxes.shape[0])
        print('Num of points: ', pcd.shape[0])

        np.save(output_prefix+'/vert.npy', pcd[:, :6])
        np.save(output_prefix+'/sem_label.npy', pcd[:, -2])
        np.save(output_prefix+'/bbox.npy', bboxes)
        np.save(output_prefix+'/shape.npy', shape_labels)
        np.save(output_prefix+'/normal.npy', normal)


if __name__=='__main__':    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)        
    all_scenes = [line.rstrip('/Annotations\n') for line in open('meta/anno_paths.txt')]
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(export_scan, args=args), all_scenes)
    else:
        for scene in all_scenes:
            export_scan(scene, args)