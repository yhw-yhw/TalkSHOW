import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.append(os.getcwd())
from glob import glob

import numpy as np
import json
import smplx as smpl

from nets import *
from repro_nets import *
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

def init_model(model_name, model_path, args, config):
    if model_name == 'freeMo':
        # generator = freeMo_Generator(args)
        # generator = freeMo_Generator(args)
        generator = freeMo_dev(args, config)
        # generator.load_state_dict(torch.load(model_path)['generator'])
    elif model_name == 'smplx_S2G':
        generator = smplx_S2G(args, config)
    elif model_name == 'StyleGestures':
        generator = StyleGesture_Generator(
            args,
            config
        )
    elif model_name == 'Audio2Gestures':
        config.Train.using_mspec_stat = False
        generator = Audio2Gesture_Generator(
            args,
            config,
            torch.zeros([1, 1, 108]),
            torch.ones([1, 1, 108])
        )
    elif model_name == 'S2G':
        generator = S2G_Generator(
            args,
            config,
        )
    elif model_name == 'Tmpt':
        generator = S2G_Generator(
            args,
            config,
        )
    else:
        raise NotImplementedError

    model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    if model_name == 'smplx_S2G':
        generator.generator.load_state_dict(model_ckpt['generator']['generator'])
    elif 'generator' in list(model_ckpt.keys()):
        generator.load_state_dict(model_ckpt['generator'])
    else:
        model_ckpt = {'generator': model_ckpt}
        generator.load_state_dict(model_ckpt)

    return generator



def prevar_loader(data_root, speakers, args, config, model_path, device, generator):
    path = model_path.split('ckpt')[0]
    file = os.path.join(os.path.dirname(path), "pre_variable.npy")
    data_base = torch_data(
        data_root=data_root,
        speakers=speakers,
        split='pre',
        limbscaling=False,
        normalization=config.Data.pose.normalization,
        norm_method=config.Data.pose.norm_method,
        split_trans_zero=False,
        num_pre_frames=config.Data.pose.pre_pose_length,
        num_generate_length=config.Data.pose.generate_length,
        num_frames=15,
        aud_feat_win_size=config.Data.aud.aud_feat_win_size,
        aud_feat_dim=config.Data.aud.aud_feat_dim,
        feat_method=config.Data.aud.feat_method,
        smplx=True,
        audio_sr=22000,
        convert_to_6d=config.Data.pose.convert_to_6d,
        expression=config.Data.pose.expression
    )

    data_base.get_dataset()
    pre_set = data_base.all_dataset
    pre_loader = data.DataLoader(pre_set, batch_size=config.DataLoader.batch_size, shuffle=False, drop_last=True)

    total_pose = []

    with torch.no_grad():
        for bat in pre_loader:
            pose = bat['poses'].to(device).to(torch.float32)
            expression = bat['expression'].to(device).to(torch.float32)
            pose = pose.permute(0, 2, 1)
            pose = torch.cat([pose[:, :15], pose[:, 15:30], pose[:, 30:45], pose[:, 45:60], pose[:, 60:]], dim=0)
            expression = expression.permute(0, 2, 1)
            expression = torch.cat([expression[:, :15], expression[:, 15:30], expression[:, 30:45], expression[:, 45:60], expression[:, 60:]], dim=0)
            pose = torch.cat([pose, expression], dim=-1)
            pose = pose.reshape(pose.shape[0], -1, 1)
            pose_code = generator.generator.pre_pose_encoder(pose).squeeze().detach().cpu()
            total_pose.append(np.asarray(pose_code))
    total_pose = np.concatenate(total_pose, axis=0)
    mean = np.mean(total_pose, axis=0)
    std = np.std(total_pose, axis=0)
    prevar = (mean, std)
    np.save(file, prevar, allow_pickle=True)

    return mean, std

def main():
    parser = parse_args()
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.cuda.set_device(device)

    config = load_JsonConfig(args.config_file)

    print('init model...')
    generator = init_model(config.Model.model_name, args.model_path, args, config)
    print('init pre-pose vectors...')
    mean, std = prevar_loader(config.Data.data_root, args.speakers, args, config, args.model_path, device, generator)

main()