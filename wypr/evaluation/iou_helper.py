# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# modifed from
# https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/iou.py
# https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_label.py

import torch
import numpy as np

def confusion_matrix(pred_ids, gt_ids, N):
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids>=0
    return np.bincount(pred_ids[idxs]*N+gt_ids[idxs], minlength=N*N).reshape((N, N)).astype(np.ulonglong)

def get_iou(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return (float(tp) / denom, tp, denom)

def evaluate_iou(pred_ids, gt_ids, dataset='scannet', verbose=True):
    if dataset == "scannet":
        VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        # Classes relabelled {-100,0,1,...,19}.
        # Predictions will all be in the set {0,1,...,19}
        CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 
                        'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 
                        'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
        UNKNOWN_ID = -100
        N_CLASSES = len(CLASS_LABELS)
    elif dataset == "scannet_dets":
        VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        # Classes relabelled {-100,0,1,...,19}.
        # Predictions will all be in the set {0,1,...,19}
        CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 
                        'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 
                        'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
        UNKNOWN_ID = -100
        N_CLASSES = len(CLASS_LABELS)
    elif dataset == "s3dis":
        raise NotImplementedError("To check the sem seg class for s3dis")
        CLASS_LABELS = ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']
        UNKNOWN_ID = -100
        N_CLASSES = len(CLASS_LABELS)
    else:
        raise ValueError("Please evaluate on the supported dataset (scannet, s3dis)")
    if verbose:
        print('evaluating', gt_ids.size, 'points on', dataset)
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    class_ious = {}
    mean_iou = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        class_ious[label_name] = get_iou(i, confusion)
        if not isinstance(class_ious[label_name], tuple): # for debug purpose
            class_ious[label_name] = (0,0,1)
        mean_iou += class_ious[label_name][0]/ N_CLASSES 
    
    if verbose:
        print('classes          IoU')
        print('----------------------------')
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            if isinstance(class_ious[label_name], tuple):
                print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]))
            else:
                print('{0:<14s}: 0.000'.format(label_name))
        print('mean IOU', mean_iou)
    return class_ious, mean_iou, confusion