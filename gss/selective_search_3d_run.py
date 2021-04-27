# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" Geometric Selective Search """
import os
import copy
import pickle
import glob
import numpy as np
import open3d as o3d

import gss.features3d as features3d
import gss.color_space_3d as color_space_3d
from wypr.dataset.scannet.vis import write_bbox

np.random.seed(1)
TAU = 0.2
WYPR_DATA_PATH = '../wypr/dataset/scannet/scannet_all_points/'
CGAL_DIR = '/data/fb_data/cgal_output_0929/'
SEG_PATH = './seg_results'

def _calc_adjacency_matrix(shapes, n_region):
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

def hierarchical_segmentation(pcd, pcd_color, scene_id, feature_mask=features3d.SimilarityMask(1, 1, 1, 1), seg=None):
    # load pre_computed
    with open(os.path.join(CGAL_DIR, scene_id+'.pkl'), 'rb') as f:
        info = pickle.load(f)
    adj_mat = info['adj_mat']; A0 = info['A0']
    cgal_txt_f = os.path.join(CGAL_DIR, scene_id+'.txt')
    with open(cgal_txt_f, 'r') as fin:
        cgal_txt = fin.readlines()

    # 2nd last row is empty line; last row are the missed points
    n_region = len(cgal_txt) - 2
    shapes = []
    # shape_id per point, initialized with dummy value n_region
    F0 = np.ones(len(pcd.points)) * n_region

    for i in range(len(cgal_txt[:-2])):
        row = cgal_txt[i]
        _points = row.rstrip(' \n').split(' ')
        row_list = [int(item) for item in _points]
        pcd_i = o3d.geometry.PointCloud()
        idx_i = np.array(row_list)
        pcd_i.points = o3d.utility.Vector3dVector(np.array(pcd.points)[idx_i])
        shapes += [pcd_i]
        F0[idx_i] = i
    
    # adj_mat, A0 = _calc_adjacency_matrix(shapes, n_region)
    feature_extractor = features3d.Features3D(pcd, pcd_color, shapes, F0, n_region, feature_mask, tau=TAU, seg=seg)
    
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

def _selective_search_one(pcd, color_formate, scene_id, similarity_weight, seg=None):
    pcd_color = np.uint8(255 * np.array(pcd.colors))
    pcd_color_converted = color_space_3d.convert_color(pcd_color, color_formate)
    (R, F, L) = hierarchical_segmentation(pcd, pcd_color_converted, scene_id, similarity_weight, seg=seg)
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

def calc_iou(box_a, box_b):
    """Computes IoU of two axis aligned bboxes.
    Args:
        box_a, box_b: 6D of center and lengths        
    Returns:
        iou
    """        
        
    max_a = box_a[0:3] + box_a[3:6]/2
    max_b = box_b[0:3] + box_b[3:6]/2    
    min_max = np.array([max_a, max_b]).min(0)
        
    min_a = box_a[0:3] - box_a[3:6]/2
    min_b = box_b[0:3] - box_b[3:6]/2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    return 1.0*intersection / union

def single_scene_precision_recall(labels, pred, iou_thresh=0.25):
    """Compute P and R for predicted bounding boxes. Ignores classes!
    Args:
        labels: (N x bbox) ground-truth bounding boxes (6 dims) 
        pred: (M x (bbox + conf)) predicted bboxes with confidence and maybe classification
    Returns:
        TP, FP, FN
    """
    # for each pred box with high conf (C), compute IoU with all gt boxes. 
    # TP = number of times IoU > th ; FP = C - TP 
    # FN - number of scene objects without good match
    gt_bboxes = labels[:, :6]      
    num_scene_bboxes = gt_bboxes.shape[0]
    
    proposal_bbox = pred[:, :6]
    num_proposal_bboxes = proposal_bbox.shape[0]
    # init an array to keep iou between generated and scene bboxes
    iou_arr = np.zeros([num_proposal_bboxes, num_scene_bboxes])    
    for g_idx in range(num_proposal_bboxes):
        for s_idx in range(num_scene_bboxes):            
            iou_arr[g_idx, s_idx] = calc_iou(proposal_bbox[g_idx ,:], gt_bboxes[s_idx, :])
    good_match_arr = (iou_arr >= iou_thresh)

    TP = good_match_arr.any(axis=0).sum()    
    recall = TP / num_scene_bboxes    
    ABO = iou_arr.max(0).mean()
    return recall, ABO

if __name__=='__main__':
    """ mask order: "size", "segmentation", "fill", "volume" """
    # names = ["fill", "volume", "sv", "fv", "sfv"]
    # masks = [features3d.SimilarityMask(0, 0, 1, 0), features3d.SimilarityMask(0, 0, 0, 1),
    #          features3d.SimilarityMask(1, 0, 0, 1), features3d.SimilarityMask(0, 0, 1, 1),
    #          features3d.SimilarityMask(1, 0, 1, 1)]

    # names = ["seg"]
    # masks = [features3d.SimilarityMask(0, 1, 0, 0)]

    # names = ["sg", "vg", "fg"]
    # masks = [features3d.SimilarityMask(1, 1, 0, 0), features3d.SimilarityMask(0, 1, 0, 1),
    #          features3d.SimilarityMask(0, 1, 1, 0)]

    # names = ["svg", "sfg"]
    # masks = [features3d.SimilarityMask(1, 1, 0, 1), features3d.SimilarityMask(1, 1, 1, 0)]

    names = ["vfg", "svfg"]
    masks = [features3d.SimilarityMask(0, 1, 1, 1), features3d.SimilarityMask(1, 1, 1, 1)]
    
    all_files = [line.rstrip() for line in open('../wypr/dataset/scannet/meta_data/scannetv2_val.txt')]
    # all_files += [line.rstrip() for line in open('../wypr/dataset/scannet/meta_data/scannetv2_train.txt')]
    for name, mask in zip(names, masks):
        out_dir = 'computed_proposal/'+ name 
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for i in range(len(all_files)):
            print(i, len(all_files))
            scene_id = all_files[i]
            output_file = os.path.join(out_dir, scene_id+'_prop.npy')
            if not os.path.isfile(output_file):
                points = np.load(os.path.join(WYPR_DATA_PATH, all_files[i]+'_vert.npy'))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255)
                # o3d.io.write_point_cloud(scene_id+"_pcd.ply", pcd)    

                if mask[1] == 1:
                    seg_file = os.path.join(SEG_PATH, scene_id+'_sem_label.npy')
                    assert os.path.exists(seg_file)
                    seg_i = np.load(seg_file)
                    proposals  = _selective_search_one(pcd, 'hsv', scene_id, mask, seg=seg_i)
                else:
                    proposals  = _selective_search_one(pcd, 'hsv', scene_id, mask)
            
                boxes = np.stack([item[1] for item in proposals])
                boxes = np.hstack((boxes, np.arange(boxes.shape[0]).reshape(-1,1)))
                # write_bbox(boxes, 'raw_cgal/'+scene_id+'_proprosals_tau%.2f_%s.ply'%(TAU, n))
                # # post-processing
                boxes_post = post_process(boxes)
                np.save(output_file, boxes_post)
                # write_bbox(boxes_post, 'raw_cgal/'+scene_id+'_proprosals_nms_post_tau%.2f_%s.ply'%(TAU, n))
                        
                # gt_box_fn = os.path.join(WYPR_DATA_PATH, scene_id+'_bbox.npy')
                # gt_boxes = np.load(gt_box_fn)
                # Rec, ABO = single_scene_precision_recall(gt_boxes, boxes_post)
                # print(name, boxes.shape, boxes_post.shape)
                # print("IoU: 0.25, recall: %.4f, ABO: %.4f" %(Rec, ABO))
                # Rec, ABO = single_scene_precision_recall(gt_boxes, boxes_post, 0.5)
                # print("IoU: 0.5, recall: %.4f, ABO: %.4f" %(Rec, ABO))
