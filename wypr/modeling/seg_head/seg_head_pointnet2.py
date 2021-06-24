# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from wypr.modeling.backbone.pointnet2.pointnet2_modules import PointnetFPModule

class Pointnet2SegHead(nn.Module):
    """ Segmentation head for pointnet++ """
    def __init__(self, input_feature_dim, num_class, suffix=""):
        super().__init__()
        self.suffix = suffix
        self.seg_fp1 = PointnetFPModule(mlp=[256+128, 256, 256])
        self.seg_fp2 = PointnetFPModule(mlp=[input_feature_dim+256, 256, 256])
        self.classifier = torch.nn.Sequential(
                            torch.nn.BatchNorm1d(256), 
                            torch.nn.ReLU(True), 
                            torch.nn.Conv1d(256, num_class, kernel_size=1))

    def forward(self, end_points=None, classification=True):
        """Forward pass of the network """
        features_1 = self.seg_fp1(
            end_points['sa1_xyz'], end_points['sa2_xyz'], 
            end_points['sa1_features'], end_points['backbone_feat'])
        features_2 = self.seg_fp2(
            end_points['input_xyz'], end_points['sa1_xyz'], 
            end_points['input_features'], features_1)
        end_points['sem_seg_feat'+self.suffix] = features_2
        if classification:
            end_points['sem_seg_pred'+self.suffix] = self.classifier(features_2)
        return end_points