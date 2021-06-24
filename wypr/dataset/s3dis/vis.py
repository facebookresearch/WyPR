# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as pyplot

from wypr.dataset.scannet.vis import write_bbox
from wypr.dataset.scannet.util import create_color_palette

def main():
    scan_name = 'Area_1/office_15'
    scene_name = 's3dis_all_points/' + scan_name
    output_folder = 'data_viz/'
    os.makedirs(output_folder, exist_ok=True)

    pt = np.load(os.path.join(scene_name, 'vert.npy'))
    seg = np.load(os.path.join(scene_name, 'sem_label.npy'))
    annot = np.load(os.path.join('s3dis_processed', scan_name+'_annot.npy'))

    colors = np.asarray(create_color_palette())
    seg_color = colors[seg.astype(np.int16)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(seg_color.astype(np.float32) / 255.)
    o3d.io.write_point_cloud(os.path.join(output_folder, scan_name.replace('/', '-') +'_sem.ply'), pcd)

    inst_color = colors[annot[:,-1].astype(np.int16)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(inst_color.astype(np.float32) / 255.)
    o3d.io.write_point_cloud(os.path.join(output_folder, scan_name.replace('/', '-') +'_inst.ply'), pcd)

    instance_bboxes = np.load(os.path.join(scene_name, 'bbox.npy'))
    output_fn = os.path.join(output_folder, scan_name.replace('/', '-') +'_det.ply')
    write_bbox(instance_bboxes, output_fn)

if __name__ == '__main__':
    main()
