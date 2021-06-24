# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" Pre-processing scannet data """
import os
import json
import datetime
import numpy as np
import open3d as o3d

from wypr.dataset.scannet.util import read_label_mapping, read_mesh_vertices_rgb


SCANNET_DIR = 'scans'
TRAIN_SCAN_NAMES = [line.rstrip() for line in open('meta_data/scannet_train.txt')]
OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39])
OUTPUT_FOLDER = './scannet_all_points'
LABEL_MAP_FILE = 'meta_data/scannetv2-labels.combined.tsv'
CGAL_DIR = 'cgal_output'

def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(mesh_file, agg_file, seg_file, meta_file, label_map_file, cgal_file, compute_normal=False):
    """ points are XYZ RGB (RGB in 0-255),
        semantic label as nyu40 ids,
        instance label as 1-#instance,
        box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """
    label_map = read_label_mapping(label_map_file,
        label_from='raw_category', label_to='nyu40id')    
    mesh_vertices = read_mesh_vertices_rgb(mesh_file)

    # load primitives
    # 2nd last row is empty line; last row are the missed points
    with open(cgal_file, 'r') as fin:  cgal_txt = fin.readlines()
    shape_labels = np.ones(mesh_vertices.shape[0], dtype=np.int64) * -1 # -1: unassigned
    for i in range(len(cgal_txt) - 2):
        _points = cgal_txt[i].rstrip(' \n').split(' ')
        idx_i = np.array([int(item) for item in _points])
        shape_labels[idx_i] = i
    assert np.sum(shape_labels==-1) == len(cgal_txt[-1].rstrip(' \n').split(' '))

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    mesh_vertices[:,0:3] = pts[:,0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_bboxes = np.zeros((num_instances,7))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids==obj_id, 0:3]
        if len(obj_pc) == 0: continue
        # Compute axis aligned box
        # An axis aligned bounding box is parameterized by
        # (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box,
        # dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:,0])
        ymin = np.min(obj_pc[:,1])
        zmin = np.min(obj_pc[:,2])
        xmax = np.max(obj_pc[:,0])
        ymax = np.max(obj_pc[:,1])
        zmax = np.max(obj_pc[:,2])
        bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2,
            xmax-xmin, ymax-ymin, zmax-zmin, label_id])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id-1,:] = bbox 

    # compute surface normal
    surface_normal = None
    if compute_normal:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(mesh_vertices[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(mesh_vertices[:, 3:] / 255)
        pcd.estimate_normals()
        surface_normal = np.asarray(pcd.normals)

    return_list = [mesh_vertices, label_ids, instance_ids, instance_bboxes, object_id_to_label_id, shape_labels]
    if compute_normal:
        return_list.append(surface_normal)    
    return return_list


def exported(prefix):
    return os.path.isfile(prefix+'_vert.npy') and os.path.isfile(prefix+'_sem_label.npy')  \
        and os.path.isfile(prefix+'_bbox.npy') and os.path.isfile(prefix+'_shape.npy') \
        and os.path.isfile(prefix+'_normal.npy')  # and os.path.isfile(prefix+'_ins_label.npy') 


def export_scan():        
    if not os.path.exists(OUTPUT_FOLDER):
        print('Creating new data folder: {}'.format(OUTPUT_FOLDER))                
        os.mkdir(OUTPUT_FOLDER)        
        
    for idx in range(len(TRAIN_SCAN_NAMES)):
        scan_name = TRAIN_SCAN_NAMES[idx]
        print(idx, len(TRAIN_SCAN_NAMES), scan_name, datetime.datetime.now())
        output_prefix = os.path.join(OUTPUT_FOLDER, scan_name) 
        if not exported(output_prefix):
            mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.ply')
            agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.aggregation.json')
            seg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
            meta_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.   
            cgal_file = os.path.join(CGAL_DIR, scan_name + '.txt') 
            mesh_vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic, shape_labels, surface_normals = \
                export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, cgal_file, compute_normal=True)

            # num_instances = len(np.unique(instance_labels))
            # print('Num of instances: ', num_instances)

            num_shapes = len(np.unique(shape_labels)) - 1
            print('Num of primitives: ', num_shapes)

            bbox_mask = np.in1d(instance_bboxes[:,-1], OBJ_CLASS_IDS)
            instance_bboxes = instance_bboxes[bbox_mask,:]
            print('Num of care instances/boxes: ', instance_bboxes.shape[0])

            N = mesh_vertices.shape[0]
            print('Num of points: ', N)

            np.save(output_prefix+'_vert.npy', mesh_vertices)
            np.save(output_prefix+'_sem_label.npy', semantic_labels)
            np.save(output_prefix+'_bbox.npy', instance_bboxes)
            np.save(output_prefix+'_shape.npy', shape_labels)
            np.save(output_prefix+'_normal.npy', surface_normals)
            # np.save(output_prefix+'_ins_label.npy', instance_labels)

if __name__=='__main__':    
    export_scan()
