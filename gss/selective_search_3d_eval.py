# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import numpy as np

from wypr.evaluation import ARCalculator
from wypr.dataset.scannet.scannet import ScannetDatasetConfig

WYPR_DATA_PATH = '../wypr/dataset/scannet/'
DATASET_CONFIG = ScannetDatasetConfig()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='val', help='evaluation data split [default: val]')
    parser.add_argument('--policy', default='size', help='evaluation strategy [default: size]')
    FLAGS = parser.parse_args()

    split_filenames = os.path.join(WYPR_DATA_PATH, 'meta_data/scannetv2_{}.txt'.format(FLAGS.split))
    with open(split_filenames, 'r') as f:  
        scan_names = f.read().splitlines()   
    ar_calculator = ARCalculator(0.25, DATASET_CONFIG.class2type) 
    all_p = []
    for i in range(len(scan_names)):
        print(i, len(scan_names), scan_names[i], FLAGS.policy)
        gt_boxes = np.load(os.path.join(WYPR_DATA_PATH, 'scannet_all_points', scan_names[i]+'_bbox.npy'))
        class_ind = [np.where(DATASET_CONFIG.nyu40ids == x)[0][0] for x in gt_boxes[:,-1]]   
        assert gt_boxes.shape[0] == len(class_ind)
        batch_gt_map_cls = [(class_ind[j], gt_boxes[j, :6]) for j in range(len(class_ind))]

        prop_i = np.load(os.path.join('computed_proposal', FLAGS.policy, scan_names[i]+'_prop.npy'))
        batch_pred_map_cls = [(ii, prop_i[j, :6])  for j in range( prop_i.shape[0])  for ii in range(18)]
        all_p += [prop_i.shape[0]]
        ar_calculator.step([batch_pred_map_cls], [batch_gt_map_cls])            

    print('-'*10, 'prop: iou_thresh: 0.25', '-'*10)
    print("avg num: %.2f" % np.mean(all_p))
    metrics_dict = ar_calculator.compute_metrics()
    for key in metrics_dict:
        print('eval %s: %f'%(key, metrics_dict[key]))
