import os
import sys


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sys.path.append(os.getcwd())

from tqdm import tqdm
from transformers import Wav2Vec2Processor

from evaluation.FGD import EmbeddingSpaceEvaluator

from evaluation.metrics import LVD

import numpy as np
import smplx as smpl

from data_utils.lower_body import part2full, poses2pred
from data_utils.utils import get_mfcc_ta
from nets import *
from nets.utils import get_path, get_dpath
from trainer.options import parse_args
from data_utils import torch_data
from trainer.config import load_JsonConfig

import torch
from torch.utils import data
from data_utils.get_j import to3d, get_joints


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
    elif model_name == 's2g_body_ae':
        generator = s2g_body_ae(
            args,
            config,
        )
    else:
        raise NotImplementedError

    model_ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    generator.load_state_dict(model_ckpt['generator'])

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


def body_loss(gt, prs):
    loss_dict = {}
    # LVD
    v_diff = LVD(gt[:, :22, :], prs[:, :, :22, :], symmetrical=False, weight=False)
    loss_dict['LVD'] = v_diff
    # Accuracy
    error = (gt - prs).norm(p=2, dim=-1).sum(dim=-1).mean()
    loss_dict['error'] = error
    # Diversity
    var = prs.var(dim=0).norm(p=2, dim=-1).sum(dim=-1).mean()
    loss_dict['diverse'] = var

    return loss_dict


def test(test_loader, generator, FGD_handler, smplx_model, config):
    print('start testing')

    am = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
    am_sr = 16000

    loss_dict = {}
    B = 2
    with torch.no_grad():
        count = 0
        for bat in tqdm(test_loader, desc="Testing......"):
            count = count + 1
            # if count == 10:
            #     break
            _, poses, exp = bat['aud_feat'].to('cuda').to(torch.float32), bat['poses'].to('cuda').to(torch.float32), \
                              bat['expression'].to('cuda').to(torch.float32)
            id = bat['speaker'].to('cuda') - 20
            betas = bat['betas'][0].to('cuda').to(torch.float64)
            poses = torch.cat([poses, exp], dim=-2).transpose(-1, -2)

            cur_wav_file = bat['aud_file'][0]

            zero_face = torch.zeros([B, poses.shape[1], 103], device='cuda')

            joints_list = []

            pred = generator.infer_on_audio(cur_wav_file,
                                            id=id,
                                            fps=30,
                                            B=B,
                                            am=am,
                                            am_sr=am_sr,
                                            frame=poses.shape[0]
                                            )
            pred = torch.tensor(pred, device='cuda')

            FGD_handler.push_samples(pred, poses)

            poses = poses.squeeze()
            poses = to3d(poses, config)

            if pred.shape[2] > 129:
                pred = pred[:, :, 103:]

            pred = torch.cat([zero_face[:, :pred.shape[1], :3], pred, zero_face[:, :pred.shape[1], 3:]], dim=-1)
            full_pred = []
            for j in range(B):
                f_pred = part2full(pred[j])
                full_pred.append(f_pred)

            for i in range(full_pred.__len__()):
                full_pred[i] = full_pred[i].unsqueeze(dim=0)
            full_pred = torch.cat(full_pred, dim=0)

            pred_joints = get_joints(smplx_model, betas, full_pred)

            poses = poses2pred(poses)
            poses = torch.cat([zero_face[0, :, :3], poses[:, 3:165], zero_face[0, :, 3:]], dim=-1)
            gt_joints = get_joints(smplx_model, betas, poses[:pred_joints.shape[1]])
            FGD_handler.push_joints(pred_joints, gt_joints)
            aud = get_mfcc_ta(cur_wav_file, fps=30, sr=16000, am='not None', encoder_choice='onset')
            FGD_handler.push_aud(torch.from_numpy(aud))

            bat_loss_dict = body_loss(gt_joints, pred_joints)

            if loss_dict:  # 非空
                for key in list(bat_loss_dict.keys()):
                    loss_dict[key] += bat_loss_dict[key]
            else:
                for key in list(bat_loss_dict.keys()):
                    loss_dict[key] = bat_loss_dict[key]
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key] / count
            print(key + '=' + str(loss_dict[key].item()))

        # MAAC = FGD_handler.get_MAAC()
        # print(MAAC)
        fgd_dist, feat_dist = FGD_handler.get_scores()
        print('fgd_dist=', fgd_dist.item())
        print('feat_dist=', feat_dist.item())
        BCscore = FGD_handler.get_BCscore()
        print('Beat consistency score=', BCscore)





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
    model_name = args.body_model_name
    # model_path = get_path(model_name, model_type)
    model_path = args.body_model_path
    generator = init_model(model_name, model_path, args, config)

    ae = init_model('s2g_body_ae', './experiments/feature_extractor.pth', args,
                    config)
    FGD_handler = EmbeddingSpaceEvaluator(ae, None, 'cuda')

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

    test(test_loader, generator, FGD_handler, smplx_model, config)


if __name__ == '__main__':
    main()
