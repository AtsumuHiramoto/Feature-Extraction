#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
# 

import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        Args:
            patience  (int):  Number of epochs with no improvement after which training will be stopped.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.save_ckpt = False
        self.stop_flag = False
        self.val_loss_min = np.Inf
        
    def __call__(self, val_loss):

        if np.isnan(val_loss) or np.isinf(val_loss):
            raise RuntimeError("Invalid loss, terminating training") 
        
        score = -val_loss

        if self.best_score is None:
            self.save_ckpt = True
            self.best_score = score
        elif score < self.best_score:
            self.save_ckpt = False
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_flag = True
        else:
            self.save_ckpt = True
            self.best_score = score
            self.counter = 0

        return self.save_ckpt, self.stop_flag