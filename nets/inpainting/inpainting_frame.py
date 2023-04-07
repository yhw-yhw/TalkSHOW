import os
import sys

import torch
from torch.optim.lr_scheduler import StepLR

sys.path.append(os.getcwd())

from nets.layers import *
from nets.base import TrainWrapperBaseClass
from nets.spg.gated_pixelcnn_v2 import GatedPixelCNN as pixelcnn
from nets.inpainting.vqvae_1d_sc import VQVAE_SC as s2g_body
from nets.spg.vqvae_1d import AudioEncoder
from nets.utils import parse_audio, denormalize
from data_utils import get_mfcc, get_melspec, get_mfcc_old, get_mfcc_psf, get_mfcc_psf_min, get_mfcc_ta
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import normalize

from data_utils.lower_body import c_index, c_index_3d, c_index_6d
from data_utils.utils import smooth_geom, get_mfcc_sepa


class TrainWrapper(TrainWrapperBaseClass):
    '''
    a wrapper receving a batch from data_utils and calculate loss
    '''

    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device(self.args.gpu)
        self.global_step = 0

        self.convert_to_6d = self.config.Data.pose.convert_to_6d
        self.expression = self.config.Data.pose.expression
        self.epoch = 0
        self.init_params()
        self.num_classes = 4
        self.audio = True
        self.composition = self.config.Model.composition
        self.bh_model = self.config.Model.bh_model

        dim, layer = 512, 5
        self.AudEnc = AudioEncoder(in_dim=64, num_hiddens=256, num_residual_layers=2, num_residual_hiddens=256).to(
            self.device)
        self.Predictor = pixelcnn(2048, dim, layer, self.num_classes, self.audio, self.bh_model).to(self.device)
        self.VQ = s2g_body(self.each_dim[1] + self.each_dim[2], embedding_dim=512, num_embeddings=config.Model.code_num,
                           num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=512).to(self.device)

        self.discriminator = None
        if self.convert_to_6d:
            self.c_index = c_index_6d
        else:
            self.c_index = c_index_3d

        super().__init__(args, config)

    def init_optimizer(self):

        print('using Adam')
        self.generator_optimizer = optim.Adam(
            self.parameters(),
            lr=self.config.Train.learning_rate.generator_learning_rate,
            betas=[0.9, 0.999]
        )

    def state_dict(self):
        model_state = {
            'AudEnc': self.AudEnc.state_dict(),
            'Predictor': self.Predictor.state_dict(),
            'VQ': self.VQ.state_dict(),
            'generator_optim': self.generator_optimizer.state_dict(),
        }
        return model_state

    def load_state_dict(self, state_dict):

        from collections import OrderedDict
        new_state_dict = OrderedDict()  # create new OrderedDict that does not contain `module.`
        for k, v in state_dict.items():
            sub_dict = OrderedDict()
            if v is not None:
                for k1, v1 in v.items():
                    name = k1.replace('module.', '')
                    sub_dict[name] = v1
            new_state_dict[k] = sub_dict
        state_dict = new_state_dict

        if 'AudEnc' in state_dict:
            self.AudEnc.load_state_dict(state_dict['AudEnc'])
        if 'Predictor' in state_dict:
            self.Predictor.load_state_dict(state_dict['Predictor'])
        if 'VQ' in state_dict:
            self.VQ.load_state_dict(state_dict['VQ'])

        if 'generator_optim' in state_dict:
            self.generator_optimizer.load_state_dict(state_dict['generator_optim'])

    def init_params(self):
        if self.config.Data.pose.convert_to_6d:
            scale = 2
        else:
            scale = 1

        global_orient = round(0 * scale)
        leye_pose = reye_pose = round(0 * scale)
        jaw_pose = round(0 * scale)
        body_pose = round((63 - 24) * scale)
        left_hand_pose = right_hand_pose = round(45 * scale)
        if self.expression:
            expression = 100
        else:
            expression = 0

        b_j = 0
        jaw_dim = jaw_pose
        b_e = b_j + jaw_dim
        eye_dim = leye_pose + reye_pose
        b_b = b_e + eye_dim
        body_dim = global_orient + body_pose
        b_h = b_b + body_dim
        hand_dim = left_hand_pose + right_hand_pose
        b_f = b_h + hand_dim
        face_dim = expression

        self.dim_list = [b_j, b_e, b_b, b_h, b_f]
        self.full_dim = jaw_dim + eye_dim + body_dim + hand_dim
        self.pose = int(self.full_dim / round(3 * scale))
        self.each_dim = [jaw_dim, eye_dim + body_dim, hand_dim, face_dim]

    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None
        loss_dict = {}

        aud, poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32)

        id = bat['speaker'].to(self.device) - 20
        # id = F.one_hot(id, self.num_classes)

        poses = poses[:, self.c_index, :]

        aud = aud.permute(0, 2, 1)
        gt_poses = poses.permute(0, 2, 1)
        mask = 1

        input_poses = gt_poses * mask

        z, enc_feats = self.VQ.encode(gt_poses=input_poses)
        audio = self.AudEnc(aud[:, :].transpose(1, 2)).unsqueeze(dim=-1)
        z = self.Predictor(z, id, audio)
        _, e_q_loss, pred_poses = self.VQ.decode(z, enc_feats)

        self.generator_optimizer.zero_grad()
        loss, loss_dict = self.get_loss(pred_poses, gt_poses, e_q_loss)
        grad = torch.nn.utils.clip_grad_norm(self.parameters(), self.config.Train.max_gradient_norm)
        loss.backward()

        if torch.isnan(grad).sum() > 0:
            print('fuck')

        loss_dict['grad'] = grad.item()
        loss_dict['ce_loss'] = loss.item()
        self.generator_optimizer.step()

        return total_loss, loss_dict

    def get_loss(self,
                 pred_poses,
                 gt_poses,
                 e_q_loss,
                 ):
        loss_dict = {}

        rec_loss = torch.mean(torch.abs(pred_poses - gt_poses))
        v_pr = pred_poses[:, 1:] - pred_poses[:, :-1]
        v_gt = gt_poses[:, 1:] - gt_poses[:, :-1]
        velocity_loss = torch.mean(torch.abs(v_pr - v_gt))

        gen_loss = rec_loss + e_q_loss + velocity_loss

        loss_dict['rec_loss'] = rec_loss
        loss_dict['velocity_loss'] = velocity_loss
        loss_dict['e_q_loss'] = e_q_loss

        return gen_loss, loss_dict

    def infer_on_audio(self, aud_fn, initial_pose=None, norm_stats=None, exp=None, var=None, w_pre=False, rand=None,
                       continuity=False, id=None, fps=15, sr=22000, B=1, am=None, am_sr=None, frame=0, **kwargs):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        assert self.args.infer, "train mode"
        self.generator.eval()
        self.g_body.eval()
        self.g_hand.eval()

        if continuity:
            aud_feat, gap = get_mfcc_sepa(aud_fn, sr=sr, fps=fps)
        else:
            aud_feat = get_mfcc_ta(aud_fn, sr=sr, fps=fps, smlpx=True, type='mfcc', am=am)
        aud_feat = aud_feat.transpose(1, 0)
        aud_feat = aud_feat[np.newaxis, ...].repeat(B, axis=0)
        aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.device)

        if id is None:
            id = torch.tensor([0]).to(self.device)
        else:
            id = id.repeat(B)

        with torch.no_grad():
            aud_feat = aud_feat.permute(0, 2, 1)
            if continuity:
                self.audioencoder.eval()
                pre_pose = {}
                pre_pose['b'] = pre_pose['h'] = None
                pre_latents, pre_audio, body_0, hand_0 = self.infer(aud_feat[:, :gap], frame, id, B, pre_pose=pre_pose)
                pre_pose['b'] = body_0[:, :, -4:].transpose(1, 2)
                pre_pose['h'] = hand_0[:, :, -4:].transpose(1, 2)
                _, _, body_1, hand_1 = self.infer(aud_feat[:, gap:], frame, id, B, pre_latents, pre_audio, pre_pose)
                body = torch.cat([body_0, body_1], dim=2)
                hand = torch.cat([hand_0, hand_1], dim=2)

            else:
                if self.audio:
                    self.audioencoder.eval()
                    audio = self.audioencoder(aud_feat.transpose(1, 2), frame_num=frame).unsqueeze(dim=-1).repeat(1, 1,
                                                                                                                  1, 2)
                    latents = self.generator.generate(id, shape=[audio.shape[2], 2], batch_size=B, aud_feat=audio)
                else:
                    latents = self.generator.generate(id, shape=[aud_feat.shape[1] // 4, 2], batch_size=B)

                body_latents = latents[..., 0]
                hand_latents = latents[..., 1]

                body, _ = self.g_body.decode(b=body_latents.shape[0], w=body_latents.shape[1], latents=body_latents)
                hand, _ = self.g_hand.decode(b=hand_latents.shape[0], w=hand_latents.shape[1], latents=hand_latents)

            pred_poses = torch.cat([body, hand], dim=1).transpose(1, 2).cpu().numpy()

        output = pred_poses

        return output

    def infer(self, aud_feat, frame, id, B, pre_latents=None, pre_audio=None, pre_pose=None):
        audio = self.audioencoder(aud_feat.transpose(1, 2), frame_num=frame).unsqueeze(dim=-1).repeat(1, 1, 1, 2)
        latents = self.generator.generate(id, shape=[audio.shape[2], 2], batch_size=B, aud_feat=audio,
                                          pre_latents=pre_latents, pre_audio=pre_audio)

        body_latents = latents[..., 0]
        hand_latents = latents[..., 1]

        body, _ = self.g_body.decode(b=body_latents.shape[0], w=body_latents.shape[1],
                                     latents=body_latents, pre_state=pre_pose['b'])
        hand, _ = self.g_hand.decode(b=hand_latents.shape[0], w=hand_latents.shape[1],
                                     latents=hand_latents, pre_state=pre_pose['h'])

        return latents, audio, body, hand

    def generate(self, aud, id, frame_num=0):

        self.AudEnc.eval()
        self.Predictor.eval()
        self.VQ.eval()
        aud_feat = aud.permute(0, 2, 1)


        audio = self.audioencoder(aud_feat.transpose(1, 2), frame_num=frame_num).unsqueeze(dim=-1).repeat(1, 1, 1,
                                                                                                          2)
        latents = self.generator.generate(id, shape=[audio.shape[2], 2], batch_size=aud.shape[0], aud_feat=audio)


        body_latents = latents[..., 0]
        hand_latents = latents[..., 1]

        body = self.g_body.decode(b=body_latents.shape[0], w=body_latents.shape[1], latents=body_latents)
        hand = self.g_hand.decode(b=hand_latents.shape[0], w=hand_latents.shape[1], latents=hand_latents)

        pred_poses = torch.cat([body, hand], dim=1).transpose(1, 2)
        return pred_poses
