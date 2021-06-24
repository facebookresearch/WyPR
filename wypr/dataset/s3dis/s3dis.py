# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for s3dis
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import copy
import numpy as np
from torch.utils.data import Dataset

from wypr.utils import pc_util
from wypr.dataset.scannet.util import rotate_aligned_boxes

MAX_NUM_OBJ = 256
MAX_NUM_PROP = 1000
MAX_NUM_POINTS = 1000000
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class S3DISDatasetConfig(object):
    def __init__(self):
        self.num_class = 10
        self.type2class = {'ceiling':0, 'floor':1, 'wall':2, 'column':3,'beam':4, 
            'window':5, 'door':6, 'table':7, 'chair':8, 'bookcase':9, 'sofa':10, 
            'board':11, 'clutter':12, 'stairs':13}
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        # self.nyu40ids = np.array([1,2,3,6,7,8,9,10,11,12])
        # self.nyu40id2class = {nyu40id: i for i,nyu40id in enumerate(list(self.nyu40ids))}

        self.num_class_semseg = 14
        self.type2class_semseg = {'ceiling':0, 'floor':1, 'wall':2, 'column':3,'beam':4, 
            'window':5, 'door':6, 'table':7, 'chair':8, 'bookcase':9, 'sofa':10, 
            'board':11, 'clutter':12, 'stairs':13}
        self.class2type_semseg = {self.type2class_semseg[t]: t for t in self.type2class_semseg}   

class S3DISDataset(Dataset):    
    def __init__(self, split_set='train', num_points=20000,
                use_color=False, use_height=False, augment=False, 
                use_normal=False, precomputed_prop="", sampling_method='random'):

        self.CONFIG = S3DISDatasetConfig()
        self.data_path = os.path.join(BASE_DIR, 's3dis_all_points')
        all_scan_names = [line.rstrip('/Annotations\n') for line in open('meta/anno_paths.txt')]
        
        if split_set=='all':            
            self.scan_names = all_scan_names
        elif split_set in ['train', 'val', 'test']:
            if split_set == 'train':
                self.scan_names = []
                for scan in all_scan_names:
                    if "Area_5" not in scan:
                        self.scan_names.append(scan)
            else:
                self.scan_names = []
                for scan in all_scan_names:
                    if "Area_5" in scan:
                        self.scan_names.append(scan)
        elif split_set=='debug':   
            self.scan_names = all_scan_names[:10]
        else:
            print('illegal split name')
            return

        num_scans = len(all_scan_names)
        print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
        
        self.split_set = split_set
        self.num_points_sampled = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal
        self.augment = augment
        self.sampling_method = sampling_method

        # proposals
        self.precomputed_prop = precomputed_prop 
        self.prop_path = os.path.join(BASE_DIR, 'proposals')
       
    def __len__(self):
        return len(self.scan_names)

    def augment_input(self, mesh_vertices, normals):
        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3] # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6] 
            point_cloud[:,3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
            pcl_color = mesh_vertices[:, 3:6]
        
        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1) 

        if self.use_normal and normals is not None:
            point_cloud = np.concatenate([point_cloud, normals], 1) 

        return point_cloud, pcl_color
      
    def __getitem__(self, idx):
        """ Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            angle_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            scan_idx: int scan index in scan_names list
            pcl_color: unused
        """
        while True:
            scan_name = self.scan_names[idx]        
            mesh_vertices = np.load(os.path.join(self.data_path, scan_name, 'vert.npy'))
            semantic_labels = np.load(os.path.join(self.data_path, scan_name, 'sem_label.npy'))
            instance_bboxes = np.load(os.path.join(self.data_path, scan_name, 'bbox.npy'))
            props = None
            if os.path.isfile(os.path.join(self.prop_path, self.precomputed_prop, scan_name+'_prop.npy')) and self.precomputed_prop != "":
                props = np.load(os.path.join(self.prop_path, self.precomputed_prop, scan_name+'_prop.npy'))
                if props.shape[0] > MAX_NUM_PROP:
                    choices = np.random.choice(props.shape[0], MAX_NUM_PROP, replace=False)
                    props = props[choices]
            shape_labels = None
            if os.path.isfile(os.path.join(self.data_path, scan_name, 'shape.npy')):
                shape_labels = np.load(os.path.join(self.data_path, scan_name, 'shape.npy'))
            normals = None
            if os.path.isfile(os.path.join(self.data_path, scan_name, 'normal.npy')):
                normals = np.load(os.path.join(self.data_path, scan_name, 'normal.npy'))
            scene_tags = np.unique(semantic_labels)

        point_cloud, pcl_color = self.augment_input(mesh_vertices, normals)
            
        # ----------------- LABELS (for evaluation purpose only) ------------------                
        # all points
        point_cloud_all = np.zeros((MAX_NUM_POINTS, point_cloud.shape[1]))
        num_points = point_cloud.shape[0]
        point_cloud_all[:num_points] = copy.deepcopy(point_cloud)

        # segmentation
        sem_seg_labels_all = np.ones(MAX_NUM_POINTS)
        sem_seg_labels_all = sem_seg_labels_all * -100 
        sem_seg_labels_all[:num_points] = copy.deepcopy(semantic_labels)
    
        # shape
        if shape_labels is not None:
            shape_labels_all = np.ones(MAX_NUM_POINTS)
            shape_labels_all = shape_labels_all * -100 
            shape_labels_all[:num_points] = copy.deepcopy(shape_labels)

        # boxes
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        num_boxes = instance_bboxes.shape[0]
        target_bboxes_mask[:num_boxes] = 1
        target_bboxes[:num_boxes] = copy.deepcopy(instance_bboxes[:,:6])
        # classes
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))                                
        target_bboxes_semcls[:num_boxes] = instance_bboxes[:,-1]  

        # proposals
        target_props = np.zeros((MAX_NUM_PROP, 6))
        target_props_mask = np.zeros((MAX_NUM_PROP))    
        if props is not None:
            target_props_mask[:props.shape[0]] = 1
            target_props[:props.shape[0],:] = copy.deepcopy(props[:,:6])

        # ------------------------------- SAMPLING ------------------------------       
        if self.num_points_sampled > 0:
            if self.sampling_method == 'rand':
                point_cloud, choices = pc_util.random_sampling(point_cloud,
                    self.num_points_sampled, return_choices=True)     
            elif self.sampling_method == 'grid':
                raise NotImplementedError
                import wypr.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
                aa, bb = cpp_subsampling.compute(point_cloud[:, :3].astype(np.float32), features=point_cloud[:, 3:].astype(np.float32), sampleDl=0.01, verbose=True)
                point_cloud = np.concatenate((aa, bb), axis=1)
                point_cloud, choices = pc_util.random_sampling(point_cloud,
                    self.num_points_sampled, return_choices=True)     
            elif self.sampling_method == 'fps':
                raise NotImplementedError("TODO: more sampling method")
                choices = furthest_point_sample(
                    torch.tensor(point_cloud, dtype=torch.float, device=torch.device('cuda')), 
                    self.num_points_sampled)
                point_cloud = point_cloud[choices]
            else:
                raise ValueError("unsupported sampling method")
            pcl_color = pcl_color[choices]
            semantic_labels = semantic_labels[choices]
            if shape_labels is not None:
                shape_labels = shape_labels[choices]
        sem_seg_labels = np.ones_like(semantic_labels)
        sem_seg_labels = sem_seg_labels * -100 # pytorch default ignore_index: -100
        for _c in self.CONFIG.nyu40ids_semseg:
            sem_seg_labels[semantic_labels == _c] = self.CONFIG.nyu40id2class_semseg[_c]

        # ------------------------------- DATA AUGMENTATION ------------------------------      
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                target_bboxes[:,0] = -1 * target_bboxes[:,0]      
                target_props[:,0] = -1 * target_props[:,0]      
                
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:,1] = -1 * point_cloud[:,1]
                target_bboxes[:,1] = -1 * target_bboxes[:,1]
                target_props[:,1] = -1 * target_props[:,1] 
            
            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)
            target_props = rotate_aligned_boxes(target_props, rot_mat)

        # ------------- DATA AUGMENTATION for consistency loss -----------------------      
        point_cloud_aug = copy.deepcopy(point_cloud)  
        target_props_aug = copy.deepcopy(target_props)  
        # Flipping along the YZ plane
        if np.random.random() > 0.8:
            point_cloud[:,0] = -1 * point_cloud[:,0]       
            target_props_aug[:,0] = -1 * target_props_aug[:,0]           
        # Flipping along the XZ plane
        if np.random.random() > 0.8:
            point_cloud[:,1] = -1 * point_cloud[:,1]
            target_props_aug[:,1] = -1 * target_props_aug[:,1]  
        # roatation
        rot_angle = (np.random.random()*np.pi/6)
        rot_mat = pc_util.rotz(rot_angle)
        point_cloud_aug[:,0:3] = np.dot(point_cloud_aug[:,0:3], np.transpose(rot_mat))
        target_props_aug = rotate_aligned_boxes(target_props_aug, rot_mat)

        # scale
        min_s = 0.8;  max_s = 2 - min_s
        # scale = np.random.rand(point_cloud.shape[0]) * (max_s - min_s) + min_s
        # point_cloud_aug[:, :3] *= scale.reshape(-1, 1)
        scale = np.random.rand() * (max_s - min_s) + min_s
        point_cloud_aug[:, :3] *= scale
        target_props_aug *= scale
        # jittering
        jitter_min = 0.95; jitter_max = 2 - jitter_min
        jitter_scale = np.random.rand(point_cloud.shape[0]) * (jitter_max - jitter_min) + jitter_min
        point_cloud_aug[:, :3] *= jitter_scale.reshape(-1, 1)
        # # dropout  
        # num_aug_points_sampled = int(0.9 * point_cloud_aug.shape[0])
        # point_cloud_aug, choices_aug = pc_util.random_sampling(point_cloud, num_aug_points_sampled, return_choices=True)  
        

        # -------------------- RETURN -----------------------      
        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['gt_boxes'] = target_bboxes.astype(np.float32)
        ret_dict['gt_boxes_cls'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['gt_boxes_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['proposals'] = target_props.astype(np.float32)
        ret_dict['proposal_mask'] = target_props_mask.astype(np.float32)
        points_rois_idx = pc_util.points_within_boxes(point_cloud, target_props)
        ret_dict['proposal_points_idx'] = points_rois_idx.astype(np.int64)
        if self.split_set != "train":
            ret_dict['num_points'] = num_points
            ret_dict['point_clouds_all'] = point_cloud_all.astype(np.float32)
            ret_dict['sem_seg_label_all'] = sem_seg_labels_all.astype(np.int64)
            ret_dict['shape_labels_all'] = shape_labels_all.astype(np.int64)
        ret_dict['point_clouds_aug'] = point_cloud_aug.astype(np.float32)
        ret_dict['proposals_aug'] = target_props_aug.astype(np.float32)
        ret_dict['sem_seg_label'] = sem_seg_labels
        ret_dict['shape_labels'] = shape_labels.astype(np.int64)
        # ret_dict['aug_idx'] = choices_aug.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        return ret_dict
        
        
if __name__=='__main__': 
    pass