# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

""" An abstract class to support basic operations """
import torch
import torch.nn as nn
import pytorch_lightning as pl

import logging
logging.getLogger("lightning").setLevel(logging.INFO)

class WyPR_Meta(pl.LightningModule):
    def __init__(self, cfg, DATASET_CONFIG):
        super().__init__()
        self.cfg = cfg
        self.DATASET_CONFIG = DATASET_CONFIG
        self._build_model()

    def _build_model(self):
        pass

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        end_points = self.forward(batch)
        # logging
        for key in end_points:
            if 'loss' in key or 'acc' in key:
                self.log(key, end_points[key], on_epoch=True, logger=True, sync_dist=True)
                if batch_idx % self.cfg.batch_interval == 0 and self.global_rank == 0:
                    logging.info('ep: %d | batch: %d | %s: %.4f' % (self.current_epoch, batch_idx, key, end_points[key]))

        return dict(loss=end_points['loss'])

    def training_epoch_end(self, training_step_outputs):
        self.log('BN_momentum', self.bnm_scheduler.lmbd(self.bnm_scheduler.last_epoch), on_epoch=True, logger=True)
        if self.global_rank == 0:
            _lr = self.trainer.optimizers[0].param_groups[0]['lr']
            logging.info(' ==> ep: %d  BN_momentum: %.4f LR: %.4f' % (self.current_epoch, self.bnm_scheduler.lmbd(self.bnm_scheduler.last_epoch), _lr))
        self.bnm_scheduler.step()

    def validation_step(self, batch, batch_idx):
        end_points = self.forward(batch)
        return_dict = {}
        for key, value in end_points.items():
            if isinstance(value, list):
                return_dict[key] = [item.cpu() for item in value]
            else:
                return_dict[key] = value.cpu()
        return return_dict

    def test_step(self, batch, batch_idx):
        end_points = self.forward(batch)
        return {key: end_points[key] for key in self.return_keys}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.lr_decay_steps, gamma=0.1)

        return [optimizer], [lr_scheduler]

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
