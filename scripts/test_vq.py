import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())

from tqdm import tqdm
from transformers import Wav2Vec2Processor

from evaluation.metrics import LVD

import numpy as np
import smplx as smpl

from data_utils.lower_body import part2full, poses2pred, c_index_3d
from nets import *
from nets.utils import get_path, get_dpath
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig

import torch
from torch.utils import data
from data_utils.get_j import to3d, get_joints
from scripts.test_body import init_model, init_dataloader


def test(test_loader, generator, config):
    print('start testing')

    loss_dict = {}
    B = 1
    with torch.no_grad():
        count = 0
        for bat in tqdm(test_loader, desc="Testing......"):
            count = count + 1
            aud, poses, exp = bat['aud_feat'].to('cuda').to(torch.float32), bat['poses'].to('cuda').to(torch.float32), \
                              bat['expression'].to('cuda').to(torch.float32)
            id = bat['speaker'].to('cuda') - 20
            betas = bat['betas'][0].to('cuda').to(torch.float64)
            poses = torch.cat([poses, exp], dim=-2).transpose(-1, -2).squeeze()
            poses = to3d(poses, config).unsqueeze(dim=0).transpose(1, 2)
            # poses = poses[:, c_index_3d, :]

            cur_wav_file = bat['aud_file'][0]

            pred = generator.infer_on_audio(cur_wav_file,
                                            initial_pose=poses,
                                            id=id,
                                            fps=30,
                                            B=B
                                            )
            pred = torch.tensor(pred, device='cuda')
            bat_loss_dict = {'capacity': (poses[:, c_index_3d, :pred.shape[0]].transpose(1,2) - pred).abs().sum(-1).mean()}

            if loss_dict:  # 非空
                for key in list(bat_loss_dict.keys()):
                    loss_dict[key] += bat_loss_dict[key]
            else:
                for key in list(bat_loss_dict.keys()):
                    loss_dict[key] = bat_loss_dict[key]
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key] / count
            print(key + '=' + str(loss_dict[key].item()))


def main():
    parser = parse_args()
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.cuda.set_device(device)

    config = load_JsonConfig(args.config_file)

    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path

    print('init dataloader...')
    test_set, test_loader, norm_stats = init_dataloader(config.Data.data_root, args.speakers, args, config)
    print('init model...')
    model_name = 's2g_body_vq'
    model_type = 'n_com_8192'
    model_path = get_path(model_name, model_type)
    generator = init_model(model_name, model_path, args, config)

    test(test_loader, generator, config)


if __name__ == '__main__':
    main()
