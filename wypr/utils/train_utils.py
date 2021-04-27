from datetime import datetime
import numpy as np
import random
import torch
import pytorch_lightning as pl

def set_seed(seed=666):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)

def get_current_lr(epoch, cfg):
    lr = cfg.learning_rate
    for i, lr_decay_epoch in enumerate(cfg.lr_decay_steps):
        if epoch >= lr_decay_epoch:
            lr *= cfg.lr_decay_rates[i]
    return lr

def adjust_learning_rate(optimizer, epoch, cfg):
    lr = get_current_lr(epoch, cfg)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def log_string(LOG_FOUT, out_str):
    LOG_FOUT.write(str(datetime.now()) + '\t')
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)