# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Example script to visualize labels in the evaluation format on the corresponding mesh.
# Inputs:
#   - predicted labels as a .txt file with one line per vertex
#   - the corresponding *_vh_clean_2.ply mesh
# Outputs a .ply with vertex colors, a different color per value in the predicted .txt file
#
# example usage: visualize_labels_on_mesh.py --pred_file [path to predicted labels file] --mesh_file [path to the *_vh_clean_2.ply mesh] --output_file [output file]

import math
import os
import inspect
import json
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as pyplot

from wypr.dataset.scannet.util import create_color_palette, print_error


def align_axis(num_verts, plydata, axis_align_matrix):
    pts = np.ones(shape=[num_verts, 4], dtype=np.float32)
    pts[:,0] = plydata['vertex'].data['x']
    pts[:,1] = plydata['vertex'].data['y']
    pts[:,2] = plydata['vertex'].data['z']
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    plydata['vertex'].data['x'] = pts[:,0]
    plydata['vertex'].data['y'] = pts[:,1]
    plydata['vertex'].data['z'] = pts[:,2]
    return plydata

def load_align_matrix(meta_file):
    """ Load scene axis alignment matrix """
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    return axis_align_matrix

def vis_rgb_align(mesh_file, output_file, meta_file):
    if not output_file.endswith('.ply'):
        print_error('output file must be a .ply file')
    axis_align_matrix = load_align_matrix(meta_file)
    with open(mesh_file, 'rb') as f:
        plydata = PlyData.read(f)
    num_verts = plydata['vertex'].count
    plydata = align_axis(num_verts, plydata, axis_align_matrix)
    plydata.write(output_file)

def vis_sem_seg_align(labels, mesh_file, output_file, meta_file):
    if not output_file.endswith('.ply'):
        print_error('output file must be a .ply file')
    colors = create_color_palette()
    num_colors = len(colors)
    axis_align_matrix = load_align_matrix(meta_file)

    with open(mesh_file, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        if num_verts != len(labels):
            print_error('#predicted labels = ' + str(len(labels)) + 'vs #mesh vertices = ' + str(num_verts))
        plydata = align_axis(num_verts, plydata, axis_align_matrix)
        # *_vh_clean_2.ply has colors already
        for i in range(num_verts):
            if labels[i] >= num_colors:
                print_error('found predicted label ' + str(labels[i]) + ' not in nyu40 label set')
            color = colors[labels[i]]
            plydata['vertex']['red'][i] = color[0]
            plydata['vertex']['green'][i] = color[1]
            plydata['vertex']['blue'][i] = color[2]
    plydata.write(output_file)

def vis_inst_seg_align(labels, mesh_file, output_file, meta_file, colormap=pyplot.cm.jet):
    if not output_file.endswith('.ply'):
        print_error('output file must be a .ply file')
    colors = create_color_palette()
    num_colors = len(colors)
    axis_align_matrix = load_align_matrix(meta_file)

    with open(mesh_file, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        if num_verts != len(labels):
            print_error('#predicted labels = ' + str(len(labels)) + 'vs #mesh vertices = ' + str(num_verts))
        plydata = align_axis(num_verts, plydata, axis_align_matrix)
        # *_vh_clean_2.ply has colors already
        for i in range(num_verts):
            color = colors[int(labels[i]%41)]
            plydata['vertex']['red'][i] = color[0]
            plydata['vertex']['green'][i] = color[1]
            plydata['vertex']['blue'][i] = color[2]
    plydata.write(output_file)

def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2] , int(color[0]*255), int(color[1]*255), int(color[2]*255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()

def write_bbox(bbox, output_file=None):
    """
    bbox: np array (n, 7), last one is instance/label id
    output_file: string
    """
    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
        
        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0,0] = 1 + t*(x*x-1)
            rot[0,1] = z*s+t*x*y
            rot[0,2] = -y*s+t*x*z
            rot[1,0] = -z*s+t*x*y
            rot[1,1] = 1+t*(y*y-1)
            rot[1,2] = x*s+t*y*z
            rot[2,0] = y*s+t*x*z
            rot[2,1] = -x*s+t*y*z
            rot[2,2] = 1+t*(z*z-1)
            return rot


        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks+1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius*math.cos(theta), radius*math.sin(theta), height*i/stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if (math.fabs(dotx) != 1.0):
                    axis = np.array([1,0,0]) - dotx * va
                else:
                    axis = np.array([0,1,0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3,3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
            
        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges

    radius = 0.02
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []
    for box in bbox:
        x_min = box[0] - box[3] / 2
        x_max = box[0] + box[3] / 2
        y_min = box[1] - box[4] / 2
        y_max = box[1] + box[4] / 2
        z_min = box[2] - box[5] / 2
        z_max = box[2] + box[5] / 2
        box_min = np.array([x_min, y_min, z_min])
        box_max = np.array([x_max, y_max, z_max])
        r, g, b = create_color_palette()[int(box[6]%41)]
        edges = get_bbox_edges(box_min, box_max)
        for k in range(len(edges)):
            cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
            cur_num_verts = len(verts)
            cyl_color = [[r/255.0,g/255.0,b/255.0] for _ in cyl_verts]
            cyl_verts = [x + offset for x in cyl_verts]
            cyl_ind = [x + cur_num_verts for x in cyl_ind]
            verts.extend(cyl_verts)
            indices.extend(cyl_ind)
            colors.extend(cyl_color)

    if output_file is None:
        return verts, colors, indices

    write_ply(verts, colors, indices, output_file)

def main():
    scan_name = 'scene0474_05'
    scene_name = 'scannet_all_points/' + scan_name
    output_folder = 'data_viz_on_mesh_dump/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    mesh_fn = os.path.join('scans', scan_name, scan_name + '_vh_clean_2.ply')
    assert os.path.isfile(mesh_fn)
    meta_fn =  os.path.join('scans', scan_name, scan_name + '.txt')

    # rgb mesh
    output_fn = os.path.join(output_folder, scan_name + '_rgb.ply')
    vis_rgb_align(mesh_fn, output_fn, meta_fn)

    # sem-seg
    semantic_labels = np.load(scene_name+'_sem_label.npy')
    output_fn = os.path.join(output_folder, scan_name + '_sem-seg.ply')
    vis_sem_seg_align(semantic_labels, mesh_fn, output_fn, meta_fn)

    # inst-seg
    instance_labels = np.load(scene_name+'_ins_label.npy')
    output_fn = os.path.join(output_folder, scan_name + '_inst-seg.ply')
    vis_inst_seg_align(instance_labels, mesh_fn, output_fn, meta_fn)

    # 3D det
    instance_bboxes = np.load(scene_name+'_bbox.npy')
    output_fn = os.path.join(output_folder, scan_name + '_3d-det.ply')
    write_bbox(instance_bboxes, output_fn)

if __name__ == '__main__':
    main()
