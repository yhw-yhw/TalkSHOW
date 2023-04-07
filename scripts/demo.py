import os
import sys
# os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())

from transformers import Wav2Vec2Processor
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
from data_utils.rotation_conversion import rotation_6d_to_matrix, matrix_to_axis_angle
from data_utils.lower_body import part2full, pred2poses, poses2pred, poses2poses
from visualise.rendering import RenderTool

global device
device = 'cpu'

def init_model(model_name, model_path, args, config):
    if model_name == 's2g_face':
        generator = s2g_face(
            args,
            config,
        )
    elif model_name == 's2g_body_vq':
        generator = s2g_body_vq(
            args,
            config,
        )
    elif model_name == 's2g_body_pixel':
        generator = s2g_body_pixel(
            args,
            config,
        )
    elif model_name == 's2g_LS3DCG':
        generator = LS3DCG(
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


def init_dataloader(data_root, speakers, args, config):
    if data_root.endswith('.csv'):
        raise NotImplementedError
    else:
        data_class = torch_data
    if 'smplx' in config.Model.model_name or 's2g' in config.Model.model_name:
        data_base = torch_data(
            data_root=data_root,
            speakers=speakers,
            split='test',
            limbscaling=False,
            normalization=config.Data.pose.normalization,
            norm_method=config.Data.pose.norm_method,
            split_trans_zero=False,
            num_pre_frames=config.Data.pose.pre_pose_length,
            num_generate_length=config.Data.pose.generate_length,
            num_frames=30,
            aud_feat_win_size=config.Data.aud.aud_feat_win_size,
            aud_feat_dim=config.Data.aud.aud_feat_dim,
            feat_method=config.Data.aud.feat_method,
            smplx=True,
            audio_sr=22000,
            convert_to_6d=config.Data.pose.convert_to_6d,
            expression=config.Data.pose.expression,
            config=config
        )
    else:
        data_base = torch_data(
            data_root=data_root,
            speakers=speakers,
            split='val',
            limbscaling=False,
            normalization=config.Data.pose.normalization,
            norm_method=config.Data.pose.norm_method,
            split_trans_zero=False,
            num_pre_frames=config.Data.pose.pre_pose_length,
            aud_feat_win_size=config.Data.aud.aud_feat_win_size,
            aud_feat_dim=config.Data.aud.aud_feat_dim,
            feat_method=config.Data.aud.feat_method
        )
    if config.Data.pose.normalization:
        norm_stats_fn = os.path.join(os.path.dirname(args.model_path), "norm_stats.npy")
        norm_stats = np.load(norm_stats_fn, allow_pickle=True)
        data_base.data_mean = norm_stats[0]
        data_base.data_std = norm_stats[1]
    else:
        norm_stats = None

    data_base.get_dataset()
    infer_set = data_base.all_dataset
    infer_loader = data.DataLoader(data_base.all_dataset, batch_size=1, shuffle=False)

    return infer_set, infer_loader, norm_stats


def get_vertices(smplx_model, betas, result_list, exp, require_pose=False):
    vertices_list = []
    poses_list = []
    expression = torch.zeros([1, 50])

    for i in result_list:
        vertices = []
        poses = []
        for j in range(i.shape[0]):
            output = smplx_model(betas=betas,
                                 expression=i[j][165:265].unsqueeze_(dim=0) if exp else expression,
                                 jaw_pose=i[j][0:3].unsqueeze_(dim=0),
                                 leye_pose=i[j][3:6].unsqueeze_(dim=0),
                                 reye_pose=i[j][6:9].unsqueeze_(dim=0),
                                 global_orient=i[j][9:12].unsqueeze_(dim=0),
                                 body_pose=i[j][12:75].unsqueeze_(dim=0),
                                 left_hand_pose=i[j][75:120].unsqueeze_(dim=0),
                                 right_hand_pose=i[j][120:165].unsqueeze_(dim=0),
                                 return_verts=True)
            vertices.append(output.vertices.detach().cpu().numpy().squeeze())
            # pose = torch.cat([output.body_pose, output.left_hand_pose, output.right_hand_pose], dim=1)
            pose = output.body_pose
            poses.append(pose.detach().cpu())
        vertices = np.asarray(vertices)
        vertices_list.append(vertices)
        poses = torch.cat(poses, dim=0)
        poses_list.append(poses)
    if require_pose:
        return vertices_list, poses_list
    else:
        return vertices_list, None


global_orient = torch.tensor([3.0747, -0.0158, -0.0152])


def infer(g_body, g_face, smplx_model, rendertool, config, args):
    betas = torch.zeros([1, 300], dtype=torch.float64).to(device)
    am = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
    am_sr = 16000
    num_sample = args.num_sample
    cur_wav_file = args.audio_file
    id = args.id
    face = args.only_face
    stand = args.stand
    if face:
        body_static = torch.zeros([1, 162], device=device)
        body_static[:, 6:9] = torch.tensor([3.0747, -0.0158, -0.0152]).reshape(1, 3).repeat(body_static.shape[0], 1)

    result_list = []

    pred_face = g_face.infer_on_audio(cur_wav_file,
                                      initial_pose=None,
                                      norm_stats=None,
                                      w_pre=False,
                                      # id=id,
                                      frame=None,
                                      am=am,
                                      am_sr=am_sr
                                      )
    pred_face = torch.tensor(pred_face).squeeze().to(device)
    # pred_face = torch.zeros([gt.shape[0], 105])

    if config.Data.pose.convert_to_6d:
        pred_jaw = pred_face[:, :6].reshape(pred_face.shape[0], -1, 6)
        pred_jaw = matrix_to_axis_angle(rotation_6d_to_matrix(pred_jaw)).reshape(pred_face.shape[0], -1)
        pred_face = pred_face[:, 6:]
    else:
        pred_jaw = pred_face[:, :3]
        pred_face = pred_face[:, 3:]

    id = torch.tensor([id], device=device)

    for i in range(num_sample):
        pred_res = g_body.infer_on_audio(cur_wav_file,
                                         initial_pose=None,
                                         norm_stats=None,
                                         txgfile=None,
                                         id=id,
                                         var=None,
                                         fps=30,
                                         w_pre=False
                                         )
        pred = torch.tensor(pred_res).squeeze().to(device)

        if pred.shape[0] < pred_face.shape[0]:
            repeat_frame = pred[-1].unsqueeze(dim=0).repeat(pred_face.shape[0] - pred.shape[0], 1)
            pred = torch.cat([pred, repeat_frame], dim=0)
        else:
            pred = pred[:pred_face.shape[0], :]

        body_or_face = False
        if pred.shape[1] < 275:
            body_or_face = True
        if config.Data.pose.convert_to_6d:
            pred = pred.reshape(pred.shape[0], -1, 6)
            pred = matrix_to_axis_angle(rotation_6d_to_matrix(pred))
            pred = pred.reshape(pred.shape[0], -1)

        if config.Model.model_name == 's2g_LS3DCG':
            pred = torch.cat([pred[:, :3], pred[:, 103:], pred[:, 3:103]], dim=-1)
        else:
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
    dict = np.concatenate(result_list[:], axis=0)
    file_name = 'visualise/video/' + config.Log.name + '/' + \
                cur_wav_file.split('\\')[-1].split('.')[-2].split('/')[-1]
    np.save(file_name, dict)

    rendertool._render_sequences(cur_wav_file, vertices_list, stand=stand, face=face, whole_body=args.whole_body)


def main():
    parser = parse_args()
    args = parser.parse_args()
    # device = torch.device(args.gpu)
    # torch.cuda.set_device(device)


    config = load_JsonConfig(args.config_file)

    face_model_name = args.face_model_name
    face_model_path = args.face_model_path
    body_model_name = args.body_model_name
    body_model_path = args.body_model_path
    smplx_path = './visualise/'

    os.environ['smplx_npz_path'] = config.smplx_npz_path
    os.environ['extra_joint_path'] = config.extra_joint_path
    os.environ['j14_regressor_path'] = config.j14_regressor_path

    print('init model...')
    generator = init_model(body_model_name, body_model_path, args, config)
    generator2 = None
    generator_face = init_model(face_model_name, face_model_path, args, config)

    print('init smlpx model...')
    dtype = torch.float64
    model_params = dict(model_path=smplx_path,
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
    smplx_model = smpl.create(**model_params).to(device)
    print('init rendertool...')
    rendertool = RenderTool('visualise/video/' + config.Log.name)

    infer(generator, generator_face, smplx_model, rendertool, config, args)


if __name__ == '__main__':
    main()
