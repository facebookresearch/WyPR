# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def get_model(cfg, DATASET_CONFIG):
    if cfg.model == 'seg_det_net_ts': 
        from wypr.modeling.models.seg_det_net import WyPR_SegDetTSNet as model
    else:
        raise ValueError("Un-supported model")

    return model(cfg, DATASET_CONFIG)
