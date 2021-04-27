# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import numpy as np
import random
from gss.selective_search_3d_run import nms_3d_faster

WYPR_DATA_PATH = '../wypr/dataset/scannet/'
MAX_NUM_PROP = 1000
POLICIES = ["size", "fill", 'volume', 'sv', 'sf', 'fv', 'sfv', 'seg_40']
SPLITS = ['val'] #['val', 'train']

ens_num = 2

if __name__=='__main__':
    iou_thresh = 0.7
    for _ in range(100):
        random.shuffle(POLICIES)
        policy = sorted(POLICIES[:ens_num])
        for split_set in SPLITS:
            output_dir = 'computed_proposal/' + '-'.join(policy)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            split_filenames = os.path.join(WYPR_DATA_PATH, 'meta_data/scannetv2_{}.txt'.format(split_set))
            with open(split_filenames, 'r') as f:  
                scan_names = f.read().splitlines()   
            for i in range(len(scan_names)):
                output_file = os.path.join(output_dir, scan_names[i]+'_prop.npy')
                print(i, len(scan_names), scan_names[i])
                if os.path.isfile(output_file):
                    continue
                props = [np.load(os.path.join( 'computed_proposal/', p, scan_names[i]+'_prop.npy')) for p in policy]
                all_props = np.vstack(props)
                all_props[:, 6] = np.random.rand(all_props.shape[0])
                pick = nms_3d_faster(all_props, iou_thresh)
                props_nms = all_props[pick]
                if props_nms.shape[0] > MAX_NUM_PROP:
                    choices = np.random.choice(props_nms.shape[0], MAX_NUM_PROP, replace=False)
                    props_nms = props_nms[choices]

                print('-'.join(policy), "before %d, after %d" % (all_props.shape[0], props_nms.shape[0]))
                np.save(output_file, props_nms)

