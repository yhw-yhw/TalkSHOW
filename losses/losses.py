import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KeypointLoss(nn.Module):
    def __init__(self):
        super(KeypointLoss, self).__init__()
    
    def forward(self, pred_seq, gt_seq, gt_conf=None):
        #pred_seq: (B, C, T)
        if gt_conf is not None:
            gt_conf = gt_conf >= 0.01
            return F.mse_loss(pred_seq[gt_conf], gt_seq[gt_conf], reduction='mean')
        else:
            return F.mse_loss(pred_seq, gt_seq)


class KLLoss(nn.Module):
    def __init__(self, kl_tolerance):
        super(KLLoss, self).__init__()
        self.kl_tolerance = kl_tolerance

    def forward(self, mu, var, mul=1):
        kl_tolerance = self.kl_tolerance * mul * var.shape[1] / 64
        kld_loss = -0.5 * torch.sum(1 + var - mu**2 - var.exp(), dim=1)
        # kld_loss = -0.5 * torch.sum(1 + (var-1) - (mu) ** 2 - (var-1).exp(), dim=1)
        if self.kl_tolerance is not None:
            # above_line = kld_loss[kld_loss > self.kl_tolerance]
            # if len(above_line) > 0:
            #     kld_loss = torch.mean(kld_loss)
            # else:
            #     kld_loss = 0
            kld_loss = torch.where(kld_loss > kl_tolerance, kld_loss, torch.tensor(kl_tolerance, device='cuda'))
        # else:
        kld_loss = torch.mean(kld_loss)
        return kld_loss


class L2KLLoss(nn.Module):
    def __init__(self, kl_tolerance):
        super(L2KLLoss, self).__init__()
        self.kl_tolerance = kl_tolerance

    def forward(self, x):
        # TODO: check
        kld_loss = torch.sum(x ** 2, dim=1)
        if self.kl_tolerance is not None:
            above_line = kld_loss[kld_loss > self.kl_tolerance]
            if len(above_line) > 0:
                kld_loss = torch.mean(kld_loss)
            else:
                kld_loss = 0
        else:
            kld_loss = torch.mean(kld_loss)
        return kld_loss

class L2RegLoss(nn.Module):
    def __init__(self):
        super(L2RegLoss, self).__init__()
    
    def forward(self, x):
        #TODO: check
        return torch.sum(x**2)


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, x):
        # TODO: check
        return torch.sum(x ** 2)


class AudioLoss(nn.Module):
    def __init__(self):
        super(AudioLoss, self).__init__()
    
    def forward(self, dynamics, gt_poses):
        #pay attention, normalized
        mean = torch.mean(gt_poses, dim=-1).unsqueeze(-1)
        gt = gt_poses - mean
        return F.mse_loss(dynamics, gt)

L1Loss = nn.L1Loss