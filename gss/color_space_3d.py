# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy
import skimage.io
import skimage.color

def convert_color(pcd_color, name):
    converters = {'rgb'  : lambda pcd_color: pcd_color,
                  'lab'  : to_Lab,
                  'rgi'  : to_rgI,
                  'hsv'  : to_HSV,
                  'nrgb' : to_nRGB,
                  'hue'  : to_Hue}

    return converters[name](pcd_color)

def to_grey(pcd_color):
    grey_img = (255 * skimage.color.rgb2grey(pcd_color)).astype(numpy.uint8)
    return numpy.dstack([grey_img, grey_img, grey_img])

def to_Lab(pcd_color):
    lab = skimage.color.rgb2lab(pcd_color)
    l = 255 * lab[:, 0] / 100    # L component ranges from 0 to 100
    a = 127 + lab[:, 1]          # a component ranges from -127 to 127
    b = 127 + lab[:, 2]          # b component ranges from -127 to 127
    return numpy.dstack([l, a, b]).astype(numpy.uint8)

def to_rgI(pcd_color):
    rgi = pcd_color.copy()
    rgi[:, 2] = to_grey(pcd_color)[:, 0]
    return rgi

def to_HSV(pcd_color):
    return (255 * skimage.color.rgb2hsv(pcd_color)).astype(numpy.uint8)

def to_nRGB(pcd_color):
    _pcd_color = pcd_color / 255.0
    norm_pcd_color = numpy.sqrt(_pcd_color[:, 0] ** 2 + _pcd_color[:, 1] ** 2 + _pcd_color[:, 2] ** 2)
    norm_r = (_pcd_color[:, 0] / norm_pcd_color * 255).astype(numpy.uint8)
    norm_g = (_pcd_color[:, 1] / norm_pcd_color * 255).astype(numpy.uint8)
    norm_b = (_pcd_color[:, 2] / norm_pcd_color * 255).astype(numpy.uint8)
    return numpy.dstack([norm_r, norm_g, norm_b])

def to_Hue(pcd_color):
    I_h = to_HSV(pcd_color)[:, 0]
    return numpy.dstack([I_h, I_h, I_h])

