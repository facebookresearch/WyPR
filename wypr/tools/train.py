# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from wypr.dataset import get_dataset
from wypr.modeling.models import get_model
from wypr.utils.train_utils import set_seed

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

@hydra.main(config_path=os.path.join("../config"), config_name="config")
def my_app(cfg : DictConfig) -> None:
    if cfg.seed != -1:
        set_seed(cfg.seed)

    # data
    DATASET_CONFIG, train_loader, val_loader = get_dataset(cfg)
    
    # model
    model = get_model(cfg, DATASET_CONFIG)
   
    # checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # monitor='loss',
        save_top_k=-1,
        period=1,
        filepath=os.path.join(os.getcwd(), '{epoch}'),
        verbose=True,
    )

    # training
    tb_logger = pl.loggers.TensorBoardLogger(os.getcwd(),  name="wypr_log")
    trainer = pl.Trainer(
        gpus=list(cfg.gpus),
        max_epochs=cfg.max_epoch,
        accelerator=cfg.distrib_backend,
        logger=tb_logger,
        weights_summary='full',
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=cfg.eval_freq,
        sync_batchnorm=True if len(list(cfg.gpus)) > 1 else False,
        resume_from_checkpoint=cfg.resume_path if cfg.resume_path != 'none' else None,
        gradient_clip_val=cfg.grad_clip if cfg.grad_clip > 0 else 0,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    my_app()