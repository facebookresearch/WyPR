# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions and class to calculate Average Precisions for 3D object detection. """

import torch
import numpy as np
from wypr.utils.eval_det import eval_det_cls, eval_det_multiprocessing, get_iou
from wypr.utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
   
def parse_groundtruths(inputs):
    """ Parse groundtruth labels to OBB parameters. 
        Args:
            end_points: dict
        Returns:
            batch_gt_map_cls: a list  of len == batch_size (BS)
                [gt_list_i], i = 0, 1, ..., BS-1
                where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
                where j = 0, ..., num of objects - 1 at sample input i
    """
    gt_boxes = inputs['gt_boxes']
    box_label_mask = inputs['gt_boxes_mask']
    sem_cls_label = inputs['gt_boxes_cls']

    bsize = gt_boxes.shape[0]
    K2 = gt_boxes.shape[1] # K2==MAX_NUM_OBJ
    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append([(sem_cls_label[i,j].item(), gt_boxes[i,j].cpu().numpy()) for j in range(K2) if box_label_mask[i,j]==1])

    return batch_gt_map_cls

def parse_proposals(end_points, config_dict):
    """ Parse proposals to OBB parameters and suppress overlapping boxes  
        Args:
            end_points: dict
            config_dict: dict
        Returns:
            batch_propos: a list of len == batch size (BS)
    """
    proposals = end_points['proposals']
    box_label_mask = end_points['proposal_mask']

    bsize = proposals.shape[0]
    K2 = proposals.shape[1] # K2==MAX_NUM_PROP

    batch_propos = []
    for i in range(bsize):
        cur_list = []
        for ii in range(config_dict['dataset_config'].num_class):
            cur_list += [(ii, proposals[i,j].numpy()) for j in range(K2) if box_label_mask[i,j] == 1]
        batch_propos.append(cur_list)

    return batch_propos

def get_3d_box(box):
    ''' box is array(x, y, z, l, w, h)
        output (8,3) array for 3D box cornders
    '''
    x, y, z, l, w, h = box
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    corners_3d = np.vstack([x_corners,y_corners,z_corners])
    corners_3d[0,:] = corners_3d[0,:] + x
    corners_3d[1,:] = corners_3d[1,:] + y
    corners_3d[2,:] = corners_3d[2,:] + z
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def parse_predictions(
        end_points, num_class, remove_empty_box=False,
        use_3d_nms=True, cls_nms=True,
        nms_iou=0.25, use_old_type_nms=False,
        conf_thresh=0):
    """ Parse predictions to OBB parameters and suppress overlapping boxes """
    proposals = end_points['proposals'].cpu().numpy()
    bsize = proposals.shape[0]
    box_label_mask = end_points['proposal_mask']
    roi_scores = [_logits.detach().cpu().numpy() for _logits in end_points['roi_logits']]
    pred_sem_cls = [np.argmax(_score, -1) for _score in roi_scores]
    nonempty_rois = end_points['non_empty_roi_idx']
    nonempty_box_mask = [np.ones(len(nonempty_rois[_i])) for _i in range(bsize)]

    # # debug
    # gt_boxes = end_points['gt_boxes']
    # box_label_mask = end_points['gt_boxes_mask']
    # sem_cls_label = end_points['gt_boxes_cls']
    # roi_scores = []
    # pred_sem_cls = [[] for _ in range(bsize)]
    # for i in range(bsize):
    #     num_rois_i = nonempty_rois[i].shape[0] 
    #     num_gt_box = int(box_label_mask[i].sum().item())
    #     gt_boxes_i = gt_boxes[i, :num_gt_box].cpu().numpy()
    #     roi_scores_j = []
    #     for j in range(num_rois_i):
    #         ious = [get_iou(proposals[i, j], _box) for _box in gt_boxes_i]
    #         _idx = np.argmax(ious)
    #         c = sem_cls_label[i, _idx].item()
    #         sc = np.repeat(np.max(ious), num_class, axis=0)
    #         roi_scores_j.append(sc)
    #         pred_sem_cls[i].append(c)
    #     roi_scores.append(np.asarray(roi_scores_j))


    pred_corners_3d = [np.ones((len(_rois), 8, 3)) for _rois in nonempty_rois]
    for i in range(bsize):
        K_i = nonempty_rois[i].shape[0]
        for j in range(K_i):
            pred_corners_3d[i][j] = get_3d_box(proposals[i, j])

    # Remove predicted boxes without any point within them..
    if remove_empty_box:
        batch_pc = end_points['point_clouds'].cpu().numpy()[:,:,0:3] # B,N,3
        for i in range(bsize):
            pc = batch_pc[i,:,:] # (N,3)
            K_i = nonempty_rois[i].shape[0]
            for j in range(K_i):
                pc_in_box, inds = extract_pc_in_box3d(pc, pred_corners_3d[i][j])
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i][j] = 0

    # NMS
    if use_3d_nms and not cls_nms:
        pred_mask = [np.zeros(len(nonempty_rois[_i])) for _i in range(bsize)]
        for i in range(bsize):
            K_i = nonempty_rois[i].shape[0]
            boxes_3d_with_prob = np.zeros((K_i, 7))
            for j in range(K_i):
                boxes_3d_with_prob[j,0] = np.min(pred_corners_3d[i][j,:,0])
                boxes_3d_with_prob[j,1] = np.min(pred_corners_3d[i][j,:,1])
                boxes_3d_with_prob[j,2] = np.min(pred_corners_3d[i][j,:,2])
                boxes_3d_with_prob[j,3] = np.max(pred_corners_3d[i][j,:,0])
                boxes_3d_with_prob[j,4] = np.max(pred_corners_3d[i][j,:,1])
                boxes_3d_with_prob[j,5] = np.max(pred_corners_3d[i][j,:,2])
                boxes_3d_with_prob[j,6] = roi_scores[i][j, pred_sem_cls[i][j]]
            nonempty_box_inds = np.where(nonempty_box_mask[i] == 1)[0]
            pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i]==1],
                nms_iou, use_old_type_nms)
            assert(len(pick)>0)
            pred_mask[i][nonempty_box_inds[pick]] = 1
    elif use_3d_nms and cls_nms:
        pred_mask = [np.zeros(len(nonempty_rois[_i])) for _i in range(bsize)]
        for i in range(bsize):
            K_i = nonempty_rois[i].shape[0]
            boxes_3d_with_prob = np.zeros((K_i, 8))
            for j in range(K_i):
                boxes_3d_with_prob[j,0] = np.min(pred_corners_3d[i][j,:,0])
                boxes_3d_with_prob[j,1] = np.min(pred_corners_3d[i][j,:,1])
                boxes_3d_with_prob[j,2] = np.min(pred_corners_3d[i][j,:,2])
                boxes_3d_with_prob[j,3] = np.max(pred_corners_3d[i][j,:,0])
                boxes_3d_with_prob[j,4] = np.max(pred_corners_3d[i][j,:,1])
                boxes_3d_with_prob[j,5] = np.max(pred_corners_3d[i][j,:,2])
                boxes_3d_with_prob[j,6] = roi_scores[i][j, pred_sem_cls[i][j]]
                boxes_3d_with_prob[j,7] = pred_sem_cls[i][j] 
            nonempty_box_inds = np.where(nonempty_box_mask[i] == 1)[0]
            pick = nms_3d_faster_samecls(
                boxes_3d_with_prob[nonempty_box_mask[i]==1],
                nms_iou, use_old_type_nms)
            assert(len(pick)>0)
            pred_mask[i][nonempty_box_inds[pick]] = 1
    else:
        pred_mask = [np.ones(len(nonempty_rois[_i])) for _i in range(bsize)]

    batch_props = []
    for i in range(bsize):
        num_rois_i = nonempty_rois[i].shape[0] 
        cur_list = []
        for j in range(num_rois_i):
            c = pred_sem_cls[i][j]
            if pred_mask[i][j] == 1 and roi_scores[i][j, c] > conf_thresh and c != num_class:
                cur_list += [(c, proposals[i, j], roi_scores[i][j, c])]
        batch_props.append(cur_list)
    return batch_props


class APCalculator(object):
    ''' Calculating Average Precision '''
    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
            Args:
                ap_iou_thresh: float between 0 and 1.0
                    IoU threshold to judge whether a prediction is positive.
                class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()
        
    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
            Args:
                batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
                batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                    should have the same length with batch_pred_map_cls (batch_size)
        """
        
        bsize = len(batch_pred_map_cls)
        assert(bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i] 
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i] 
            self.scan_cnt += 1
    
    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh, get_iou_func=get_iou)
        ret_dict = {} 
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s AP'%(clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall'%(clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall'%(clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {} # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {} # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0
