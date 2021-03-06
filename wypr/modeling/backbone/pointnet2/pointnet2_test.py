# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

''' Testing customized ops. '''
import torch
import numpy as np
from wypr.modeling.backbone.pointnet2.pointnet2_utils import three_interpolate

def test_interpolation_grad():
    batch_size = 1
    feat_dim = 2
    m = 4
    feats = torch.randn(batch_size, feat_dim, m, requires_grad=True).float().cuda()
    
    def interpolate_func(inputs):
        idx = torch.from_numpy(np.array([[[0,1,2],[1,2,3]]])).int().cuda()
        weight = torch.from_numpy(np.array([[[1,1,1],[2,2,2]]])).float().cuda()
        interpolated_feats = three_interpolate(inputs, idx, weight)
        return interpolated_feats
    
    assert (torch.autograd.gradcheck(interpolate_func, feats, atol=1e-1, rtol=1e-1))

if __name__=='__main__':
    test_interpolation_grad()
