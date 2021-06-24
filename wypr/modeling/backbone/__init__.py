# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def get_backbone(name):
    if name == "pointnet2":
        from wypr.modeling.backbone.backbone_pointnet2 import Pointnet2Backbone
        return Pointnet2Backbone
    elif name == 'sparseconv':
        raise NotImplementedError
    else:
        raise ValueError('unsupoorted backbone')