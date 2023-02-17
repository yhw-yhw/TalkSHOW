'''
Warning: metrics are for reference only, may have limited significance
'''
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch

from data_utils.lower_body import rearrange, symmetry
import torch.nn.functional as F

def data_driven_baselines(gt_kps):
    '''
    gt_kps: T, D
    '''
    gt_velocity = np.abs(gt_kps[1:] - gt_kps[:-1])
    
    mean= np.mean(gt_velocity, axis=0)[np.newaxis] #(1, D)
    mean = np.mean(np.abs(gt_velocity-mean))
    last_step = gt_kps[1] - gt_kps[0]
    last_step = last_step[np.newaxis] #(1, D)
    last_step = np.mean(np.abs(gt_velocity-last_step))
    return last_step, mean

def Batch_LVD(gt_kps, pr_kps, symmetrical, weight):
    if gt_kps.shape[0] > pr_kps.shape[1]:
        length = pr_kps.shape[1]
    else:
        length = gt_kps.shape[0]
    gt_kps = gt_kps[:length]
    pr_kps = pr_kps[:, :length]
    global symmetry
    symmetry = torch.tensor(symmetry).bool()

    if symmetrical:
        # rearrange for compute symmetric. ns means non-symmetrical joints, ys means symmetrical joints.
        gt_kps = gt_kps[:, rearrange]
        ns_gt_kps = gt_kps[:, ~symmetry]
        ys_gt_kps = gt_kps[:, symmetry]
        ys_gt_kps = ys_gt_kps.reshape(ys_gt_kps.shape[0], -1, 2, 3)
        ns_gt_velocity = (ns_gt_kps[1:] - ns_gt_kps[:-1]).norm(p=2, dim=-1)
        ys_gt_velocity = (ys_gt_kps[1:] - ys_gt_kps[:-1]).norm(p=2, dim=-1)
        left_gt_vel = ys_gt_velocity[:, :, 0].sum(dim=-1)
        right_gt_vel = ys_gt_velocity[:, :, 1].sum(dim=-1)
        move_side = torch.where(left_gt_vel>right_gt_vel, torch.ones(left_gt_vel.shape).cuda(),  torch.zeros(left_gt_vel.shape).cuda())
        ys_gt_velocity = torch.mul(ys_gt_velocity[:, :, 0].transpose(0,1), move_side) + torch.mul(ys_gt_velocity[:, :, 1].transpose(0,1), ~move_side.bool())
        ys_gt_velocity = ys_gt_velocity.transpose(0,1)
        gt_velocity = torch.cat([ns_gt_velocity, ys_gt_velocity], dim=1)

        pr_kps = pr_kps[:, :, rearrange]
        ns_pr_kps = pr_kps[:, :, ~symmetry]
        ys_pr_kps = pr_kps[:, :, symmetry]
        ys_pr_kps = ys_pr_kps.reshape(ys_pr_kps.shape[0], ys_pr_kps.shape[1], -1, 2, 3)
        ns_pr_velocity = (ns_pr_kps[:, 1:] - ns_pr_kps[:, :-1]).norm(p=2, dim=-1)
        ys_pr_velocity = (ys_pr_kps[:, 1:] - ys_pr_kps[:, :-1]).norm(p=2, dim=-1)
        left_pr_vel = ys_pr_velocity[:, :, :, 0].sum(dim=-1)
        right_pr_vel = ys_pr_velocity[:, :, :, 1].sum(dim=-1)
        move_side = torch.where(left_pr_vel > right_pr_vel, torch.ones(left_pr_vel.shape).cuda(),
                                torch.zeros(left_pr_vel.shape).cuda())
        ys_pr_velocity = torch.mul(ys_pr_velocity[..., 0].permute(2, 0, 1), move_side) + torch.mul(
            ys_pr_velocity[..., 1].permute(2, 0, 1), ~move_side.long())
        ys_pr_velocity = ys_pr_velocity.permute(1, 2, 0)
        pr_velocity = torch.cat([ns_pr_velocity, ys_pr_velocity], dim=2)
    else:
        gt_velocity = (gt_kps[1:] - gt_kps[:-1]).norm(p=2, dim=-1)
        pr_velocity = (pr_kps[:, 1:] - pr_kps[:, :-1]).norm(p=2, dim=-1)

    if weight:
        w = F.softmax(gt_velocity.sum(dim=1).normal_(), dim=0)
    else:
        w = 1 / gt_velocity.shape[0]

    v_diff = ((pr_velocity - gt_velocity).abs().sum(dim=-1) * w).sum(dim=-1).mean()

    return v_diff


def LVD(gt_kps, pr_kps, symmetrical=False, weight=False):
    gt_kps = gt_kps.squeeze()
    pr_kps = pr_kps.squeeze()
    if len(pr_kps.shape) == 4:
        return Batch_LVD(gt_kps, pr_kps, symmetrical, weight)
    # length = np.minimum(gt_kps.shape[0], pr_kps.shape[0])
    length = gt_kps.shape[0]-10
    # gt_kps = gt_kps[25:length]
    # pr_kps = pr_kps[25:length] #(T, D)
    # if pr_kps.shape[0] < gt_kps.shape[0]:
    #     pr_kps = np.pad(pr_kps, [[0, int(gt_kps.shape[0]-pr_kps.shape[0])], [0, 0]], mode='constant')
    
    gt_velocity = (gt_kps[1:] - gt_kps[:-1]).norm(p=2, dim=-1)
    pr_velocity = (pr_kps[1:] - pr_kps[:-1]).norm(p=2, dim=-1)
    
    return (pr_velocity-gt_velocity).abs().sum(dim=-1).mean()

def diversity(kps):
    '''
    kps: bs, seq, dim
    '''
    dis_list = []
    #the distance between each pair
    for i in range(kps.shape[0]):
        for j in range(i+1, kps.shape[0]):
            seq_i = kps[i]
            seq_j = kps[j]
            
            dis = np.mean(np.abs(seq_i - seq_j))
            dis_list.append(dis)
    return np.mean(dis_list)
