# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def get_segmentation_head(cfg, input_feature_dim, num_class, model, role="none"):
    suffix = "_t" if role == "teacher" else ""
    if model == 'pointnet2-vanilla':
        from wypr.modeling.seg_head.seg_head_pointnet2 import Pointnet2SegHead
        return Pointnet2SegHead(input_feature_dim, num_class, suffix)
    else:
        return ValueError('un-supported segmentation head')