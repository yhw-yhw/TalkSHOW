import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.getcwd())

from tqdm import tqdm
from transformers import Wav2Vec2Processor

from evaluation.metrics import LVD

import numpy as np
import smplx as smpl

from nets import *
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig
from data_utils.get_j import get_joints

import torch
from torch.utils import data


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

    if config.Data.pose.normalization:
        norm_stats_fn = os.path.join(os.path.dirname(args.model_path), "norm_stats.npy")
        norm_stats = np.load(norm_stats_fn, allow_pickle=True)
        data_base.data_mean = norm_stats[0]
        data_base.data_std = norm_stats[1]
    else:
        norm_stats = None

    data_base.get_dataset()
    test_set = data_base.all_dataset
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False)

    return test_set, test_loader, norm_stats


def face_loss(gt, gt_param, pr, pr_param):
    loss_dict = {}

    jaw_xyz = gt[:, 22:25, :] - pr[:, 22:25, :]
    jaw_dist = jaw_xyz.norm(p=2, dim=-1)
    jaw_dist = jaw_dist.sum(dim=-1).mean()
    loss_dict['jaw_l1'] = jaw_dist

    landmark_xyz = gt[:, 74:] - pr[:, 74:]
    landmark_dist = landmark_xyz.norm(p=2, dim=-1)
    landmark_dist = landmark_dist.sum(dim=-1).mean()
    loss_dict['landmark_l1'] = landmark_dist

    face_gt = torch.cat([gt[:, 22:25], gt[:, 74:]], dim=1)
    face_pr = torch.cat([pr[:, 22:25], pr[:, 74:]], dim=1)

    loss_dict['LVD'] = LVD(face_gt, face_pr, symmetrical=False, weight=False)

    return loss_dict


def test(test_loader, generator, smplx_model, args, config):
    print('start testing')

    am = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
    am_sr = 16000

    loss_dict = {}
    with torch.no_grad():
        i = 0
        for bat in tqdm(test_loader, desc="Testing......"):
            i = i + 1
            aud, poses, exp = bat['aud_feat'].to('cuda').to(torch.float32), bat['poses'].to('cuda').to(torch.float32), \
                              bat['expression'].to('cuda').to(torch.float32)
            id = bat['speaker'].to('cuda') - 20
            betas = bat['betas'][0].to('cuda').to(torch.float64)
            poses = torch.cat([poses, exp], dim=-2).transpose(-1, -2).squeeze()
            # poses = to3d(poses, config)

            cur_wav_file = bat['aud_file'][0]
            pred_face = generator.infer_on_audio(cur_wav_file,
                                                 id=id,
                                                 frame=poses.shape[0],
                                                 am=am,
                                                 am_sr=am_sr
                                                 )

            pred_face = torch.tensor(pred_face).to('cuda').squeeze()
            if pred_face.shape[1] > 103:
                pred_face = pred_face[:, :103]
            zero_poses = torch.zeros([pred_face.shape[0], 162], device='cuda')

            full_param = torch.cat([pred_face[:, :3], zero_poses, pred_face[:, 3:]], dim=-1)

            poses[:, 3:165] = full_param[:, 3:165]
            gt_joints = get_joints(smplx_model, betas, poses)
            pred_joints = get_joints(smplx_model, betas, full_param)
            bat_loss_dict = face_loss(gt_joints, poses, pred_joints, full_param)

            if loss_dict:  # 非空
                for key in list(bat_loss_dict.keys()):
                    loss_dict[key] += bat_loss_dict[key]
            else:
                for key in list(bat_loss_dict.keys()):
                    loss_dict[key] = bat_loss_dict[key]
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key] / i
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
    face_model_name = args.face_model_name
    face_model_path = args.face_model_path
    generator_face = init_model(face_model_name, face_model_path, args, config)

    print('init smlpx model...')
    dtype = torch.float64
    smplx_path = './visualise/'
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
                        dtype=dtype, )
    smplx_model = smpl.create(**model_params).to('cuda')

    test(test_loader, generator_face, smplx_model, args, config)


if __name__ == '__main__':
    main()
