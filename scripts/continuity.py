import os
import sys
# os.environ["PYOPENGL_PLATFORM"] = "egl"
from transformers import Wav2Vec2Processor
from visualise.rendering import RenderTool

sys.path.append(os.getcwd())
from glob import glob

import numpy as np
import json
import smplx as smpl

from nets import *
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from scripts.diversity import init_model, init_dataloader, get_vertices
from data_utils.lower_body import part2full, pred2poses, poses2pred, poses2poses
from data_utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle
import time


global_orient = torch.tensor([3.0747, -0.0158, -0.0152])


def infer(data_root, g_body, g_face, g_body2, exp_name, infer_loader, infer_set, device, norm_stats, smplx,
          smplx_model, rendertool, args=None, config=None, var=None):
    am = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
    am_sr = 16000
    num_sample = 1
    face = False
    if face:
        body_static = torch.zeros([1, 162], device='cuda')
        body_static[:, 6:9] = torch.tensor([3.0747, -0.0158, -0.0152]).reshape(1, 3).repeat(body_static.shape[0], 1)
    stand = False
    j = 0
    gt_0 = None

    for bat in infer_loader:
        poses_ = bat['poses'].to(torch.float32).to(device)
        if poses_.shape[-1] == 300:
            j = j + 1
            if j > 1000:
                continue
            id = bat['speaker'].to('cuda') - 20
            if config.Data.pose.expression:
                expression = bat['expression'].to(device).to(torch.float32)
                poses = torch.cat([poses_, expression], dim=1)
            else:
                poses = poses_
            cur_wav_file = bat['aud_file'][0]
            betas = bat['betas'][0].to(torch.float64).to('cuda')
            # betas = torch.zeros([1, 300], dtype=torch.float64).to('cuda')
            gt = poses.to('cuda').squeeze().transpose(1, 0)
            if config.Data.pose.normalization:
                gt = denormalize(gt, norm_stats[0], norm_stats[1]).squeeze(dim=0)
            if config.Data.pose.convert_to_6d:
                if config.Data.pose.expression:
                    gt_exp = gt[:, -100:]
                    gt = gt[:, :-100]

                gt = gt.reshape(gt.shape[0], -1, 6)
                gt = matrix_to_axis_angle(rotation_6d_to_matrix(gt)).reshape(gt.shape[0], -1)
                gt = torch.cat([gt, gt_exp], -1)
            if face:
                gt = torch.cat([gt[:, :3], body_static.repeat(gt.shape[0], 1), gt[:, -100:]], dim=-1)

            result_list = [gt]

            # cur_wav_file = '.\\training_data\\french-V4.wav'

            # pred_face = g_face.infer_on_audio(cur_wav_file,
            #                                   initial_pose=poses_,
            #                                   norm_stats=None,
            #                                   w_pre=False,
            #                                   # id=id,
            #                                   frame=None,
            #                                   am=am,
            #                                   am_sr=am_sr
            #                                   )
            #
            # pred_face = torch.tensor(pred_face).squeeze().to('cuda')

            pred_face = torch.zeros([gt.shape[0], 103], device='cuda')
            pred_jaw = pred_face[:, :3]
            pred_face = pred_face[:, 3:]

            # id = torch.tensor([0], device='cuda')

            for i in range(num_sample):
                pred_res = g_body.infer_on_audio(cur_wav_file,
                                                 initial_pose=poses_,
                                                 norm_stats=norm_stats,
                                                 txgfile=None,
                                                 id=id,
                                                 var=var,
                                                 fps=30,
                                                 continuity=True,
                                                 smooth=False
                                                 )
                pred = torch.tensor(pred_res).squeeze().to('cuda')

                if pred.shape[0] < pred_face.shape[0]:
                    repeat_frame = pred[-1].unsqueeze(dim=0).repeat(pred_face.shape[0] - pred.shape[0], 1)
                    pred = torch.cat([pred, repeat_frame], dim=0)
                else:
                    pred = pred[:pred_face.shape[0], :]

                if config.Data.pose.convert_to_6d:
                    pred = pred.reshape(pred.shape[0], -1, 6)
                    pred = matrix_to_axis_angle(rotation_6d_to_matrix(pred))
                    pred = pred.reshape(pred.shape[0], -1)

                pred = torch.cat([pred_jaw, pred, pred_face], dim=-1)
                # pred[:, 9:12] = global_orient
                pred = part2full(pred, stand)
                if face:
                    pred = torch.cat([pred[:, :3], body_static.repeat(pred.shape[0], 1), pred[:, -100:]], dim=-1)
                # result_list[0] = poses2pred(result_list[0], stand)
                # if gt_0 is None:
                #     gt_0 = gt
                # pred = pred2poses(pred, gt_0)
                # result_list[0] = poses2poses(result_list[0], gt_0)

                result_list.append(pred)

            vertices_list, _ = get_vertices(smplx_model, betas, result_list, config.Data.pose.expression)

            result_list = [res.to('cpu') for res in result_list]
            dict = np.concatenate(result_list[1:], axis=0)
            file_name = 'visualise/video/' + config.Log.name + '/' + \
                        cur_wav_file.split('\\')[-1].split('.')[-2].split('/')[-1]
            np.save(file_name, dict)

            rendertool._render_continuity(cur_wav_file, vertices_list[1], frame=60)


def main():
    parser = parse_args()
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.cuda.set_device(device)

    config = load_JsonConfig(args.config_file)

    smplx = True

    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path

    print('init model...')
    body_model_name = 's2g_body_pixel'
    body_model_path = './experiments/2022-12-31-smplx_S2G-body-pixel-conti-wide/ckpt-99.pth'  # './experiments/2022-10-09-smplx_S2G-body-pixel-aud-3p/ckpt-99.pth'
    generator = init_model(body_model_name, body_model_path, args, config)

    # face_model_name = 's2g_face'
    # face_model_path = './experiments/2022-10-15-smplx_S2G-face-sgd-3p-wv2/ckpt-99.pth'  # './experiments/2022-09-28-smplx_S2G-face-faceformer-3d/ckpt-99.pth'
    # generator_face = init_model(face_model_name, face_model_path, args, config)
    generator_face = None
    print('init dataloader...')
    infer_set, infer_loader, norm_stats = init_dataloader(config.Data.data_root, args.speakers, args, config)

    print('init smlpx model...')
    dtype = torch.float64
    model_params = dict(model_path='E:/PycharmProjects/Motion-Projects/models',
                        model_type='smplx',
                        create_global_orient=True,
                        create_body_pose=True,
                        create_betas=True,
                        num_betas=300,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        use_pca=False,
                        flat_hand_mean=False,
                        create_expression=True,
                        num_expression_coeffs=100,
                        num_pca_comps=12,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        # gender='ne',
                        dtype=dtype, )
    smplx_model = smpl.create(**model_params).to('cuda')
    print('init rendertool...')
    rendertool = RenderTool('visualise/video/' + config.Log.name)

    infer(config.Data.data_root, generator, generator_face, None, args.exp_name, infer_loader, infer_set, device,
          norm_stats, smplx, smplx_model, rendertool, args, config, (None, None))


if __name__ == '__main__':
    main()
