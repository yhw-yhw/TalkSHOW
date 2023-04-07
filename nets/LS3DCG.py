'''
not exactly the same as the official repo but the results are good
'''
import sys
import os

from data_utils.lower_body import c_index_3d, c_index_6d

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

from nets.base import TrainWrapperBaseClass
from nets.layers import SeqEncoder1D
from losses import KeypointLoss, L1Loss, KLLoss
from data_utils.utils import get_melspec, get_mfcc_psf, get_mfcc_ta
from nets.utils import denormalize

class Conv1d_tf(nn.Conv1d):
    """
    Conv1d with the padding behavior from TF
    modified from https://github.com/mlperf/inference/blob/482f6a3beb7af2fb0bd2d91d6185d5e71c22c55f/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py
    """

    def __init__(self, *args, **kwargs):
        super(Conv1d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "same")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv1d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        if rows_odd:
            input = F.pad(input, [0, rows_odd])

        return F.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


def ConvNormRelu(in_channels, out_channels, type='1d', downsample=False, k=None, s=None, norm='bn', padding='valid'):
    if k is None and s is None:
        if not downsample:
            k = 3
            s = 1
        else:
            k = 4
            s = 2

    if type == '1d':
        conv_block = Conv1d_tf(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
        if norm == 'bn':
            norm_block = nn.BatchNorm1d(out_channels)
        elif norm == 'ln':
            norm_block = nn.LayerNorm(out_channels)
    elif type == '2d':
        conv_block = Conv2d_tf(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
        norm_block = nn.BatchNorm2d(out_channels)
    else:
        assert False

    return nn.Sequential(
        conv_block,
        norm_block,
        nn.LeakyReLU(0.2, True)
    )

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()
        self.up1 = nn.Sequential(
            ConvNormRelu(in_ch // 2 + in_ch, in_ch // 2),
            ConvNormRelu(in_ch // 2, in_ch // 2),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.up2 = nn.Sequential(
            ConvNormRelu(in_ch // 4 + in_ch // 2, in_ch // 4),
            ConvNormRelu(in_ch // 4, in_ch // 4),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.up3 = nn.Sequential(
            ConvNormRelu(in_ch // 8 + in_ch // 4, in_ch // 8),
            ConvNormRelu(in_ch // 8, in_ch // 8),
            nn.Conv1d(in_ch // 8, out_ch, 1, 1)
        )

    def forward(self, x, x1, x2, x3):
        x = F.interpolate(x, x3.shape[2])
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x)
        x = F.interpolate(x, x2.shape[2])
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = F.interpolate(x, x1.shape[2])
        x = torch.cat([x, x1], dim=1)
        x = self.up3(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, n_frames, each_dim):
        super().__init__()
        self.n_frames = n_frames

        self.down1 = nn.Sequential(
            ConvNormRelu(64, 64, '1d', False),
            ConvNormRelu(64, 128, '1d', False),
        )
        self.down2 = nn.Sequential(
            ConvNormRelu(128, 128, '1d', False),
            ConvNormRelu(128, 256, '1d', False),
        )
        self.down3 = nn.Sequential(
            ConvNormRelu(256, 256, '1d', False),
            ConvNormRelu(256, 512, '1d', False),
        )
        self.down4 = nn.Sequential(
            ConvNormRelu(512, 512, '1d', False),
            ConvNormRelu(512, 1024, '1d', False),
        )

        self.down = nn.MaxPool1d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.face_decoder = Decoder(1024, each_dim[0] + each_dim[3])
        self.body_decoder = Decoder(1024, each_dim[1])
        self.hand_decoder = Decoder(1024, each_dim[2])

    def forward(self, spectrogram, time_steps=None):
        if time_steps is None:
            time_steps = self.n_frames

        x1 = self.down1(spectrogram)
        x = self.down(x1)
        x2 = self.down2(x)
        x = self.down(x2)
        x3 = self.down3(x)
        x = self.down(x3)
        x = self.down4(x)
        x = self.up(x)

        face = self.face_decoder(x, x1, x2, x3)
        body = self.body_decoder(x, x1, x2, x3)
        hand = self.hand_decoder(x, x1, x2, x3)

        return face, body, hand


class Generator(nn.Module):
    def __init__(self,
                 each_dim,
                 training=False,
                 device=None
                 ):
        super().__init__()

        self.training = training
        self.device = device

        self.encoderdecoder = EncoderDecoder(15, each_dim)

    def forward(self, in_spec, time_steps=None):
        if time_steps is not None:
            self.gen_length = time_steps

        face, body, hand = self.encoderdecoder(in_spec)
        out = torch.cat([face, body, hand], dim=1)
        out = out.transpose(1, 2)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormRelu(input_dim, 128, '1d'),
            ConvNormRelu(128, 256, '1d'),
            nn.MaxPool1d(kernel_size=2),
            ConvNormRelu(256, 256, '1d'),
            ConvNormRelu(256, 512, '1d'),
            nn.MaxPool1d(kernel_size=2),
            ConvNormRelu(512, 512, '1d'),
            ConvNormRelu(512, 1024, '1d'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(1024, 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        out = self.net(x)
        return out


class TrainWrapper(TrainWrapperBaseClass):
    def __init__(self, args, config) -> None:
        self.args = args
        self.config = config
        self.device = torch.device(self.args.gpu)
        self.global_step = 0
        self.convert_to_6d = self.config.Data.pose.convert_to_6d
        self.init_params()

        self.generator = Generator(
            each_dim=self.each_dim,
            training=not self.args.infer,
            device=self.device,
        ).to(self.device)
        self.discriminator = Discriminator(
            input_dim=self.each_dim[1] + self.each_dim[2] + 64
        ).to(self.device)
        if self.convert_to_6d:
            self.c_index = c_index_6d
        else:
            self.c_index = c_index_3d
        self.MSELoss = KeypointLoss().to(self.device)
        self.L1Loss = L1Loss().to(self.device)
        super().__init__(args, config)

    def init_params(self):
        scale = 1

        global_orient = round(0 * scale)
        leye_pose = reye_pose = round(0 * scale)
        jaw_pose = round(3 * scale)
        body_pose = round((63 - 24) * scale)
        left_hand_pose = right_hand_pose = round(45 * scale)

        expression = 100

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
        assert (not self.args.infer), "infer mode"
        self.global_step += 1

        loss_dict = {}

        aud, poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32)
        expression = bat['expression'].to(self.device).to(torch.float32)
        jaw = poses[:, :3, :]
        poses = poses[:, self.c_index, :]

        pred = self.generator(in_spec=aud)

        D_loss, D_loss_dict = self.get_loss(
            pred_poses=pred.detach(),
            gt_poses=poses,
            aud=aud,
            mode='training_D',
        )

        self.discriminator_optimizer.zero_grad()
        D_loss.backward()
        self.discriminator_optimizer.step()

        G_loss, G_loss_dict = self.get_loss(
            pred_poses=pred,
            gt_poses=poses,
            aud=aud,
            expression=expression,
            jaw=jaw,
            mode='training_G',
        )
        self.generator_optimizer.zero_grad()
        G_loss.backward()
        self.generator_optimizer.step()

        total_loss = None
        loss_dict = {}
        for key in list(D_loss_dict.keys()) + list(G_loss_dict.keys()):
            loss_dict[key] = G_loss_dict.get(key, 0) + D_loss_dict.get(key, 0)

        return total_loss, loss_dict

    def get_loss(self,
                 pred_poses,
                 gt_poses,
                 aud=None,
                 jaw=None,
                 expression=None,
                 mode='training_G',
                 ):
        loss_dict = {}
        aud = aud.transpose(1, 2)
        gt_poses = gt_poses.transpose(1, 2)
        gt_aud = torch.cat([gt_poses, aud], dim=2)
        pred_aud = torch.cat([pred_poses[:, :, 103:], aud], dim=2)

        if mode == 'training_D':
            dis_real = self.discriminator(gt_aud)
            dis_fake = self.discriminator(pred_aud)
            dis_error = self.MSELoss(torch.ones_like(dis_real).to(self.device), dis_real) + self.MSELoss(
                torch.zeros_like(dis_fake).to(self.device), dis_fake)
            loss_dict['dis'] = dis_error

            return dis_error, loss_dict
        elif mode == 'training_G':
            jaw_loss = self.L1Loss(pred_poses[:, :, :3], jaw.transpose(1, 2))
            face_loss = self.MSELoss(pred_poses[:, :, 3:103], expression.transpose(1, 2))
            body_loss = self.L1Loss(pred_poses[:, :, 103:142], gt_poses[:, :, :39])
            hand_loss = self.L1Loss(pred_poses[:, :, 142:], gt_poses[:, :, 39:])
            l1_loss = jaw_loss + face_loss + body_loss + hand_loss

            dis_output = self.discriminator(pred_aud)
            gen_error = self.MSELoss(torch.ones_like(dis_output).to(self.device), dis_output)
            gen_loss = self.config.Train.weights.keypoint_loss_weight * l1_loss + self.config.Train.weights.gan_loss_weight * gen_error

            loss_dict['gen'] = gen_error
            loss_dict['jaw_loss'] = jaw_loss
            loss_dict['face_loss'] = face_loss
            loss_dict['body_loss'] = body_loss
            loss_dict['hand_loss'] = hand_loss
            return gen_loss, loss_dict
        else:
            raise ValueError(mode)

    def infer_on_audio(self, aud_fn, fps=30, initial_pose=None, norm_stats=None, id=None, B=1, **kwargs):
        output = []
        assert self.args.infer, "train mode"
        self.generator.eval()

        if self.config.Data.pose.normalization:
            assert norm_stats is not None
            data_mean = norm_stats[0]
            data_std = norm_stats[1]

        pre_length = self.config.Data.pose.pre_pose_length
        generate_length = self.config.Data.pose.generate_length
        # assert pre_length == initial_pose.shape[-1]
        # pre_poses = initial_pose.permute(0, 2, 1).to(self.device).to(torch.float32)
        # B = pre_poses.shape[0]

        aud_feat = get_mfcc_ta(aud_fn, sr=22000, fps=fps, smlpx=True, type='mfcc').transpose(1, 0)
        num_poses_to_generate = aud_feat.shape[-1]
        aud_feat = aud_feat[np.newaxis, ...].repeat(B, axis=0)
        aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred_poses = self.generator(aud_feat)
            pred_poses = pred_poses.cpu().numpy()
        output = pred_poses.squeeze()

        return output

    def generate(self, aud, id):
        self.generator.eval()
        pred_poses = self.generator(aud)
        return pred_poses


if __name__ == '__main__':
    from trainer.options import parse_args

    parser = parse_args()
    args = parser.parse_args(
        ['--exp_name', '0', '--data_root', '0', '--speakers', '0', '--pre_pose_length', '4', '--generate_length', '64',
         '--infer'])

    generator = TrainWrapper(args)

    aud_fn = '../sample_audio/jon.wav'
    initial_pose = torch.randn(64, 108, 4)
    norm_stats = (np.random.randn(108), np.random.randn(108))
    output = generator.infer_on_audio(aud_fn, initial_pose, norm_stats)

    print(output.shape)
