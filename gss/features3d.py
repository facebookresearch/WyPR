# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import copy
import collections
import numpy as np 
import open3d as o3d
import scipy.ndimage.filters
 
SimilarityMask = collections.namedtuple("SimilarityMask", ["size", "seg", "fill", "volume"])


class Features3D:
    def __init__(self, pcd, pcd_color, shape_pcd, label, n_region, 
                similarity_weight=SimilarityMask(1, 1, 1, 0), tau=0.02, 
                seg=None):
        self.seg = seg
        self.n_region = n_region
        self.pcd = pcd
        self.pcd_color = pcd_color
        self.shape_pcd = {i : shape_pcd[i] for i in range(self.n_region)}

        self.label = label
        self.w = similarity_weight
        # point jitter to avoid absolute flat regions
        self.tau = 0.02

        # different size metrics
        self.size    = self.__init_size()
        self.volume  = self.__init_volume()
        # self.surface = self.__init_area()
        self.seg     = self.__init_seg() if seg is not None else seg
        # self.normal = self.__init_normal()

        self.bbox    = self.__init_bounding_box()
        self.pcdsize  = self.pcd.get_axis_aligned_bounding_box().volume()
        _mesh = self.pcd.compute_convex_hull()[0]
        self.pcdvolume  = _mesh.get_volume() if _mesh.is_watertight() else self.pcdsize 

    def __init_volume(self):
        volumes = []
        for _, pcd_i in self.shape_pcd.items():
            _pcd = copy.deepcopy(pcd_i)
            _pcd.points = o3d.utility.Vector3dVector(np.asarray(_pcd.points) * (1 - self.tau/2 + np.random.rand(len(_pcd.points), 3)*self.tau) )
            _mesh = _pcd.compute_convex_hull()[0]
            _vol = _mesh.get_volume() if _mesh.is_watertight() else _pcd.get_axis_aligned_bounding_box().volume()
            volumes += [_vol]
        return {i : volumes[i] for i in range(self.n_region)}

    def __init_size(self):
        volumes = []
        for _, pcd_i in self.shape_pcd.items():
            _pcd = copy.deepcopy(pcd_i)
            _pcd.points = o3d.utility.Vector3dVector(np.asarray(_pcd.points) * (1 - self.tau/2 + np.random.rand(len(_pcd.points), 3)*self.tau)  )
            volumes += [_pcd.get_axis_aligned_bounding_box().volume()]
        return {i : volumes[i] for i in range(self.n_region)}

    def __init_area(self):
        areas = []
        for _, pcd_i in self.shape_pcd.items():
            _pcd = copy.deepcopy(pcd_i)
            _pcd.points = o3d.utility.Vector3dVector(np.asarray(_pcd.points) * (1 - self.tau/2 + np.random.rand(len(_pcd.points), 3)*self.tau)  )
            areas += [_pcd.compute_convex_hull()[0].get_surface_area()]
        return {i : areas[i] for i in range(self.n_region)}

    def __init_seg(self):
        num_classes = int(self.seg.max() + 1)
        bins_seg = range(num_classes + 1)
        bins_label = range(self.n_region + 1)
        bins = [bins_label, bins_seg]
        hist = np.histogram2d(self.label, self.seg, bins=bins)[0] #shape=(n_region, n_bin)
        l1_norm = np.sum(hist, axis=1).reshape((self.n_region, 1))
        hist = np.nan_to_num(hist / l1_norm)
        return {i : hist[i] for i in range(self.n_region)}

    def __init_color(self):
        n_bin = 25
        bin_width = int(math.ceil(255.0 / n_bin))
        bins_color = [i * bin_width for i in range(n_bin + 1)]
        bins_label = range(self.n_region + 1)
        bins = [bins_label, bins_color]
        r_hist = np.histogram2d(self.label, self.pcd_color[:, 0], bins=bins)[0] #shape=(n_region, n_bin)
        g_hist = np.histogram2d(self.label, self.pcd_color[:, 1], bins=bins)[0]
        b_hist = np.histogram2d(self.label, self.pcd_color[:, 2], bins=bins)[0]
        hist = np.hstack([r_hist, g_hist, b_hist])
        l1_norm = np.sum(hist, axis=1).reshape((self.n_region, 1))
        hist = np.nan_to_num(hist / l1_norm)
        return {i : hist[i] for i in range(self.n_region)}

    def __init_bounding_box(self):
        bbox = dict()
        for region in range(self.n_region):
            bbox[region] = self.shape_pcd[region].get_axis_aligned_bounding_box()
        return bbox

    def __calc_gradient_histogram(self, label, gaussian, n_region, nbins_orientation = 8, nbins_inten = 10):
        op = np.array([[-1, 0, 1]], dtype=np.float32)
        h = scipy.ndimage.filters.convolve(gaussian, op)
        v = scipy.ndimage.filters.convolve(gaussian, op.transpose())
        g = np.arctan2(v, h)

        # define each axis for texture histogram
        bin_width = 2 * math.pi / 8
        bins_label = range(n_region + 1)
        bins_angle = np.linspace(-math.pi, math.pi, nbins_orientation + 1)
        bins_inten = np.linspace(.0, 1., nbins_inten + 1)
        bins = [bins_label, bins_angle, bins_inten]

        # calculate 3 dimensional histogram
        ar = np.vstack([label.ravel(), g.ravel(), gaussian.ravel()]).transpose()
        hist = np.histogramdd(ar, bins = bins)[0]

        # orientation_wise intensity histograms are serialized for each region
        return np.reshape(hist, (n_region, nbins_orientation * nbins_inten))

    def __sim_size(self, i, j):
        return 1. - (self.size[i] + self.size[j]) / self.pcdsize

    def __sim_volume(self, i, j):
        return 1. - (self.volume[i] + self.volume[j]) / self.pcdvolume

    def __calc_histogram_intersection(self, vec1, vec2):
        return np.sum(np.minimum(vec1, vec2))

    def __sim_normal(self, i, j):
        return self.__calc_histogram_intersection(self.normal[i], self.normal[j])

    def __sim_seg(self, i, j):
        return self.__calc_histogram_intersection(self.seg[i], self.seg[j])

    def __sim_color(self, i, j):
        return self.__calc_histogram_intersection(self.color[i], self.color[j])

    def __sim_fill(self, i, j):
        pcd_ij = o3d.geometry.PointCloud()
        points_i = np.array(self.shape_pcd[i].points)
        points_j = np.array(self.shape_pcd[j].points)
        pcd_ij.points = o3d.utility.Vector3dVector(np.vstack((points_i, points_j)))
        bij_size = pcd_ij.get_axis_aligned_bounding_box().volume()
        return 1. - (bij_size - self.size[i] - self.size[j]) / self.pcdsize

    def similarity(self, i, j):
        sim = 0
        if self.w.size != 0:
            sim += self.w.size * self.__sim_size(i, j)
        if self.w.seg != 0:
            sim += self.w.seg * self.__sim_seg(i, j) 
        if self.w.fill != 0:
            sim += self.w.fill * self.__sim_fill(i, j)
        if self.w.volume != 0:
            sim += self.w.volume * self.__sim_volume(i, j) 
        return sim
               

    def __merge_shape_pcd(self, i, j, new_region_id):
        pcd_ij = o3d.geometry.PointCloud()
        points_i = np.array(self.shape_pcd[i].points)
        points_j = np.array(self.shape_pcd[j].points)
        pcd_ij.points = o3d.utility.Vector3dVector(np.vstack((points_i, points_j)))
        self.shape_pcd[new_region_id] = pcd_ij

    def __merge_size(self, new_region_id):
        self.size[new_region_id] = self.bbox[new_region_id].volume()

    def __merge_volume(self, new_region_id):
        _mesh = self.shape_pcd[new_region_id].compute_convex_hull()[0]
        self.volume[new_region_id] = _mesh.get_volume() if _mesh.is_watertight() else self.shape_pcd[new_region_id].get_axis_aligned_bounding_box().volume()

    def __merge_surface(self, new_region_id):
        self.surface[new_region_id] = self.shape_pcd[new_region_id].compute_convex_hull()[0].get_surface_area()

    def __histogram_merge(self, vec1, vec2, w1, w2):
        return (w1 * vec1 + w2 * vec2) / (w1 + w2)

    def __merge_color(self, i, j, new_region_id):
        self.color[new_region_id] = self.__histogram_merge(self.color[i], self.color[j], self.size[i], self.size[j])

    def __merge_seg(self, i, j, new_region_id):
        self.seg[new_region_id] = self.__histogram_merge(self.seg[i], self.seg[j], self.size[i], self.size[j])

    def __merge_normal(self, i, j, new_region_id):
        self.normal[new_region_id] = self.__histogram_merge(self.normal[i], self.normal[j], self.size[i], self.size[j])

    def __merge_bbox(self, new_region_id):
        self.bbox[new_region_id] = self.shape_pcd[new_region_id].get_axis_aligned_bounding_box()

    def merge(self, i, j):
        new_region_id = len(self.size)
        self.__merge_shape_pcd(i, j, new_region_id)
        # self.__merge_color(i, j, new_region_id)
        # self.__merge_normal(i, j, new_region_id)
        # self.__merge_surface(new_region_id)
        if self.seg is not None:
            self.__merge_seg(i, j, new_region_id)
        self.__merge_bbox(new_region_id)
        self.__merge_size(new_region_id)
        self.__merge_volume(new_region_id)
        return new_region_id

