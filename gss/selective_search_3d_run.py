# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" Geometric Selective Search """
import os
import numpy as np
import open3d as o3d
import argparse
from multiprocessing import Pool
from functools import partial

from gss.features3d import SimilarityMask
from gss.utils import _selective_search_one, post_process
from wypr.dataset.scannet.vis import write_bbox

np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='val', 
                        help='evaluation data split: train | val | trainval')
parser.add_argument('--dataset', default='scannet', 
                        help='dataset to use: scannet | s3dis')
parser.add_argument('--data_path', default='../wypr/dataset/scannet/', 
                        help='path to scannet')
parser.add_argument('--cgal_path', default='../wypr/dataset/scannet/cgal_output', 
                        help='path to cgal shape detection res')
parser.add_argument('--seg_path', default=None, 
                        help='path to segmentation res')
parser.add_argument('--tau', type=float, default=0.2,
                        help='threshold tau')
parser.add_argument('--n_proc', '-n', type=int, default=0, 
                        help='number of processes to use.')
FLAGS = parser.parse_args()


def _run_gss(args, scene_id, visualize=False):
    for name, mask in zip(args['names'], args['masks']):
        out_dir = os.path.join('computed_proposal_' + FLAGS.dataset, name)
        if FLAGS.dataset == 'scannet':
            os.makedirs(out_dir, exist_ok=True)
            output_file = os.path.join(out_dir, scene_id+'_prop.npy')
            input_file = os.path.join(FLAGS.data_path, FLAGS.dataset+'_all_points', scene_id+'_vert.npy')
        else:
            os.makedirs(os.path.join(out_dir, scene_id), exist_ok=True)
            output_file = os.path.join(out_dir, scene_id, 'prop.npy')
            input_file = os.path.join(FLAGS.data_path, FLAGS.dataset+'_all_points', scene_id, 'vert.npy')

        if not os.path.isfile(output_file):
            print(scene_id)
            points = np.load(input_file)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255)
            if visualize:
                o3d.io.write_point_cloud(os.path.join(out_dir, scene_id, "pcd.ply"), pcd)    

            # use segmentation
            if mask[1] == 1:
                seg_file = os.path.join(FLAGS.seg_path, scene_id+'_sem_pred.npy') \
                           if FLAGS.dataset == 'scannet' \
                           else os.path.join(FLAGS.seg_path, scene_id, 'sem_label.npy')
                seg_i = np.load(seg_file)
                proposals = _selective_search_one(pcd, scene_id, mask, FLAGS, seg=seg_i)
            else:
                proposals = _selective_search_one(pcd, scene_id, mask, FLAGS)
        
            boxes = np.stack([item[1] for item in proposals])
            boxes = np.hstack((boxes, np.arange(boxes.shape[0]).reshape(-1, 1)))
            if visualize:
                write_bbox(boxes, os.path.join(out_dir, scene_id, 'props_raw.ply'))

            # post-processing
            boxes_post = post_process(boxes)
            np.save(output_file, boxes_post)
            print('saved to %s' % output_file)
            if visualize:
                write_bbox(boxes_post, os.path.join(out_dir, scene_id, 'props_post.ply'))


def run_scannet(names, masks):
    all_files = [line.rstrip() for line in 
        open(os.path.join(FLAGS.data_path, 'meta_data/scannetv2_%s.txt'%FLAGS.split))]
    args = {'names': names, 'masks': masks}
    if FLAGS.n_proc != 0:
        with Pool(FLAGS.n_proc) as p:
            p.map(partial(_run_gss, args), all_files)
    else:
        for scene_id in all_files:
            _run_gss(args, scene_id)


def run_s3dis(names, masks):
    all_scan_names = [line.rstrip('/Annotations\n') for line in 
                open(os.path.join(FLAGS.data_path, 'meta/anno_paths.txt'))]
    if FLAGS.split == 'trainval':            
        scan_names = all_scan_names
    elif FLAGS.split in ['train', 'val']:
        if FLAGS.split == 'train':
            scan_names = []
            for scan in all_scan_names:
                if "Area_5" not in scan:
                    scan_names.append(scan)
        else:
            scan_names = []
            for scan in all_scan_names:
                if "Area_5" in scan:
                    scan_names.append(scan)
    else:
        raise ValueError('illegal split name')

    args = {'names': names, 'masks': masks}
    if FLAGS.n_proc != 0:
        with Pool(FLAGS.n_proc) as p:
            p.map(partial(_run_gss, args), scan_names)
    else:
        for scene_id in scan_names:
            _run_gss(args, scene_id)


if __name__=='__main__':
    """ mask order: "size", "segmentation", "fill", "volume" """
    names = ["size"]
    masks = [SimilarityMask(1, 0, 0, 0)]

    # names = ["fill", 'volume', 'sf', 'sfv']
    # masks = [SimilarityMask(0, 0, 1, 0), SimilarityMask(0, 0, 0, 1),
    #         SimilarityMask(1, 0, 1, 0), SimilarityMask(1, 0, 1, 1)]
      
    if FLAGS.dataset == 'scannet':
       run_scannet(names, masks)
    elif FLAGS.dataset == 's3dis':
       run_s3dis(names, masks)
    else:
        raise ValueError('unsupported dataset name')