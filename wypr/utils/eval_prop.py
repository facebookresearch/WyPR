# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Generic Code for Proposal Evaluation
    Input:
    For each class:
        For each image:
            Predictions: box
            Groundtruths: box
    Output:
    For each class:
        ABO,
        recal 
    
    Ref: https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/datasets/voc_eval.py
"""
import os
import sys
import numpy as np
from multiprocessing import Pool

from wypr.utils.metric_util import calc_iou # axis-aligned 3D box IoU
from wypr.utils.eval_det import get_iou, get_iou_obb, get_iou_main
from wypr.utils.box_util import box3d_iou

def eval_prop_cls(pred, gt, ovthresh=0.25, get_iou_func=get_iou):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: scalar, iou threshold
        Output:
            rec: numpy array of length nd
            ABO: Average Best Overlap (ABO)
    """

    # construct gt objects
    class_recs = {} # {img_id: {'bbox': bbox list, 'det': matched list, 'max_overlap': for MABO}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        max_overlap = [0] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det, 'max_overlap': max_overlap}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': [], 'max_overlap': []}
    
    # construct dets
    image_ids = []
    BB = []
    for img_id in pred.keys():
        for box in pred[img_id]:
            image_ids.append(img_id)
            BB.append(box)
    BB = np.array(BB) # (nd,4 or 8,3 or 6)

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        #if d%100==0: print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d,...].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        #print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
                if ovmax > R['max_overlap'][jmax]:
                    if ovmax > 1:
                        print('wrong ovmax: %f'%ovmax)
                    R['max_overlap'][jmax] = ovmax
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # # avoid divide by zero in case the first detection matches a difficult ground truth
    # prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    max_overlaps = []
    for i in range(len(class_recs)):
        if class_recs[i]['bbox'].size != 0:
            max_overlaps += class_recs[i]['max_overlap']

    assert len(max_overlaps) == npos
    ABO = np.mean(max_overlaps)

    return rec, ABO

def eval_prop_cls_wrapper(arguments):
    pred, gt, ovthresh, get_iou_func = arguments
    rec, ABO = eval_prop_cls(pred, gt, ovthresh, get_iou_func)
    return (rec, ABO)

def eval_prop_multiprocessing(pred_all, gt_all, ovthresh=0.25, get_iou_func=get_iou):
    """ Generic functions to compute precision/recall for object detection for multiple classes.
        Input:
            pred_all: map of {img_id: (classname, bbox)}
            gt_all:   map of {img_id: (classname, bbox)}
            ovthresh: scalar, iou threshold
        Output:
            rec: {classname: rec}
            ABO: {classname: prec_all}
    """
    pred = {}; gt = {} # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:  pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:  gt[classname][img_id] = []
            pred[classname][img_id].append(bbox)
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]: gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}; ABO = {}
    p = Pool(processes=10)
    ret_values = p.map(eval_prop_cls_wrapper, [(pred[classname], gt[classname], ovthresh, get_iou_func) for classname in gt.keys() if classname in pred])
    p.close()
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], ABO[classname] = ret_values[i]
        else:
            rec[classname] = 0; ABO[classname] = 0
    
    return rec, ABO