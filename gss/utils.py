# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" Geometric Selective Search """
import os
import copy
import pickle
import numpy as np
import open3d as o3d

import gss.features3d as features3d
import gss.color_space_3d as color_space_3d

def _new_adjacency_dict(A, i, j, t):
    Ak = copy.deepcopy(A)
    Ak[t] = (Ak[i] | Ak[j]) - {i, j}
    del Ak[i], Ak[j]
    for (p, Q) in Ak.items():
        if i in Q or j in Q:
            Q -= {i, j}
            Q.add(t)
    return Ak

def _merge_similarity_set(feature_extractor, Ak, S, i, j, t):
    # remove entries which have i or j
    S = list(filter(lambda x: not(i in x[1] or j in x[1]), S))

    # calculate similarity between region t and its adjacencies
    St = [(feature_extractor.similarity(t, x), (t, x)) for x in Ak[t] if t < x] +\
         [(feature_extractor.similarity(x, t), (x, t)) for x in Ak[t] if x < t]

    return sorted(S + St)

def _build_initial_similarity_set(A0, feature_extractor):
    S = list()
    for (i, J) in A0.items():
        S += [(feature_extractor.similarity(i, j), (i, j)) for j in J if i < j]

    return sorted(S)

def _new_label_image(F, i, j, t):
    Fk = np.copy(F)
    Fk[Fk == i] = Fk[Fk == j] = t
    return Fk

def hierarchical_segmentation(pcd, pcd_color, scene_id, FLAGS,
    feature_mask=features3d.SimilarityMask(1, 1, 1, 1), seg=None):

    # load pre_computed
    with open(os.path.join(FLAGS.cgal_path, scene_id+'.pkl'), 'rb') as f:
        info = pickle.load(f)
    adj_mat = info['adj_mat']; A0 = info['A0']
    
    F0 = np.load(os.path.join(FLAGS.cgal_path, scene_id+'_shape.npy'))
    n_region = len(np.unique(F0))
    shapes = []
    for i in range(n_region):
        pcd_i = o3d.geometry.PointCloud()
        idx_i = np.where(F0 == i)
        pcd_i.points = o3d.utility.Vector3dVector(np.array(pcd.points)[idx_i])
        shapes += [pcd_i]

    feature_extractor = features3d.Features3D(pcd, pcd_color, shapes, F0, n_region, feature_mask, tau=FLAGS.tau, seg=seg)
    
    # stores list of regions sorted by their similarity
    S = _build_initial_similarity_set(A0, feature_extractor)

    # stores region label and its parent (empty if initial).
    R = {i : () for i in range(n_region)}

    A = [A0]    # stores adjacency relation for each step
    F = [F0]    # stores label pcd for each step

    # greedy hierarchical grouping loop
    while len(S):
        (s, (i, j)) = S.pop()
        t = feature_extractor.merge(i, j)
        # record merged region (larger region should come first)
        R[t] = (i, j) if feature_extractor.size[j] < feature_extractor.size[i] else (j, i)
        Ak = _new_adjacency_dict(A[-1], i, j, t)
        A.append(Ak)
        S = _merge_similarity_set(feature_extractor, Ak, S, i, j, t)
        F.append(_new_label_image(F[-1], i, j, t))

    # bounding boxes for each hierarchy
    L = feature_extractor.bbox
    return (R, F, L)

def _generate_regions(R, L):
    n_ini = sum(not parent for parent in R.values())
    n_all = len(R)
    regions = list()
    for label in R.keys():
        if label >= n_ini:
            vi = np.random.rand() * label
            # center (xyz) + extent
            center_i = np.array(L[label].get_center())
            extent_i = np.array(L[label].get_extent())
            regions += [(vi, np.hstack((center_i, extent_i)) )]
    return sorted(regions)

def _selective_search_one(pcd, scene_id, similarity_weight, FLAGS, 
        seg=None, color_formate='hsv'):
    pcd_color = np.uint8(255 * np.array(pcd.colors))
    pcd_color_converted = color_space_3d.convert_color(pcd_color, color_formate)
    (R, F, L) = hierarchical_segmentation(
            pcd, pcd_color_converted, scene_id, 
            FLAGS, similarity_weight, seg=seg)
    return _generate_regions(R, L)

def nms_3d_faster(boxes, overlap_threshold, old_type=False):
    x1 = boxes[:,0] - boxes[:,3] / 2
    y1 = boxes[:,1] - boxes[:,4] / 2
    z1 = boxes[:,2] - boxes[:,5] / 2
    x2 = boxes[:,3] + boxes[:,3] / 2
    y2 = boxes[:,4] + boxes[:,4] / 2
    z2 = boxes[:,5] + boxes[:,5] / 2
    score = boxes[:,6]
    area = (x2-x1)*(y2-y1)*(z2-z1)

    I = np.argsort(score)[::-1]
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])

        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)

        if old_type:
            o = (l*w*h)/area[I[:last-1]]
        else:
            inter = l*w*h
            o = inter / (area[i] + area[I[:last-1]] - inter)

        I = np.delete(I, np.concatenate(([last-1], np.where(o>overlap_threshold)[0])))

    return pick

def post_process(boxes, iou_thresh=0.75):
    """ NMS and removing the largest boxes """
    pick = nms_3d_faster(boxes, iou_thresh)
    boxes = boxes[pick]
    # remove largest
    areas = boxes[:,3] * boxes[:,4] * boxes[:,5]
    idx = np.argmax(areas)
    boxes = np.delete(boxes, idx, 0)
    return boxes
