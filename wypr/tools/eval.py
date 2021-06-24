# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import hydra
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from wypr.dataset import get_dataset
from wypr.modeling.models import get_model
from wypr.evaluation import evaluate_iou
from wypr.utils.train_utils import log_string, set_seed


@torch.no_grad()
def evaluate_multi_pass(trainer, model, eval_loader, cfg, LOG_FOUT):
    return_key = trainer.test(model, test_dataloaders=eval_loader)
    for pass_idx in range(1, cfg.num_eval_pass):
        set_seed(666 + pass_idx)
        return_key += trainer.test(model, test_dataloaders=eval_loader)

    # Evaluate mIoU for semantic segmentation
    log_string(LOG_FOUT, '---- eval seg multi pass----')
    sem_seg_pred = np.mean(np.vstack([_pred['sem_seg_pred'][None, :, :] for _pred in return_key]), axis=0)
    sem_seg_gt = return_key[0]['sem_seg_gt']
    class_ious, miou, confusion = evaluate_iou(sem_seg_pred.argmax(axis=0), sem_seg_gt)
    for key in class_ious:
        log_string(LOG_FOUT, 'eval IoU %s: %f'%(key, class_ious[key][0]))
    log_string(LOG_FOUT, 'eval mIoU: %f'% miou)


@hydra.main(config_path=os.path.join("../config"), config_name="config")
def my_app(cfg : DictConfig) -> None:
    set_seed(666)
    assert os.path.isfile(cfg.checkpoint_path), cfg.checkpoint_path
    BASE_DIR = os.path.join(os.getcwd(), cfg.output_dir)
    LOG_DIR = os.path.join(BASE_DIR, 'log')
    DUMP_DIR = os.path.join(BASE_DIR, 'dump')
    for DIR in [BASE_DIR, LOG_DIR, DUMP_DIR]:
        os.makedirs(DIR, exist_ok=True)
    fname = 'log_eval_on_train.txt' if cfg.eval_on_train else ('log_eval_multi.txt' if cfg.multi_pass_eval else 'log_eval.txt')
    LOG_FOUT = open(os.path.join(LOG_DIR, fname), 'w')
    LOG_FOUT.write(OmegaConf.to_yaml(cfg))

    # Get dataset and model 
    DATASET_CONFIG, train_loader, eval_loader = get_dataset(cfg)
    model = get_model(cfg, DATASET_CONFIG)

    # Loading
    checkpoint = torch.load(cfg.checkpoint_path)
    key = 'model_state_dict' if 'model_state_dict' in checkpoint.keys() else 'state_dict' 
    model.load_state_dict(checkpoint[key])
    log_string(LOG_FOUT, "Loaded checkpoint %s" % cfg.checkpoint_path)

    trainer = pl.Trainer(gpus=list(cfg.gpus), accelerator='dp')

    if cfg.multi_pass_eval:
        evaluate_multi_pass(trainer, model, eval_loader, cfg, LOG_FOUT)
    else:
        trainer.test(model, test_dataloaders=eval_loader)

if __name__ == "__main__":
    my_app()