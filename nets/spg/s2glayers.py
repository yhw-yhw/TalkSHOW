'''
not exactly the same as the official repo but the results are good
'''
import sys
import os

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nets.layers import SeqEncoder1D, SeqTranslator1D

""" from https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context.git """


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    from https://github.com/mlperf/inference/blob/482f6a3beb7af2fb0bd2d91d6185d5e71c22c55f/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

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
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv1d_tf(nn.Conv1d):
    """
    Conv1d with the padding behavior from TF
    modified from https://github.com/mlperf/inference/blob/482f6a3beb7af2fb0bd2d91d6185d5e71c22c55f/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py
    """

    def __init__(self, *args, **kwargs):
        super(Conv1d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding")

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
        # if self.padding == "valid":
        #     return F.conv1d(
        #         input,
        #         self.weight,
        #         self.bias,
        #         self.stride,
        #         padding=0,
        #         dilation=self.dilation,
        #         groups=self.groups,
        #     )
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


def ConvNormRelu(in_channels, out_channels, type='1d', downsample=False, k=None, s=None, padding='valid', groups=1,
                 nonlinear='lrelu', bn='bn'):
    if k is None and s is None:
        if not downsample:
            k = 3
            s = 1
            padding = 'same'
        else:
            k = 4
            s = 2
            padding = 'valid'

    if type == '1d':
        conv_block = Conv1d_tf(in_channels, out_channels, kernel_size=k, stride=s, padding=padding, groups=groups)
        norm_block = nn.BatchNorm1d(out_channels)
    elif type == '2d':
        conv_block = Conv2d_tf(in_channels, out_channels, kernel_size=k, stride=s, padding=padding, groups=groups)
        norm_block = nn.BatchNorm2d(out_channels)
    else:
        assert False
    if bn != 'bn':
        if bn == 'gn':
            norm_block = nn.GroupNorm(1, out_channels)
        elif bn == 'ln':
            norm_block = nn.LayerNorm(out_channels)
        else:
            norm_block = nn.Identity()
    if nonlinear == 'lrelu':
        nlinear = nn.LeakyReLU(0.2, True)
    elif nonlinear == 'tanh':
        nlinear = nn.Tanh()
    elif nonlinear == 'none':
        nlinear = nn.Identity()

    return nn.Sequential(
        conv_block,
        norm_block,
        nlinear
    )


class UnetUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnetUp, self).__init__()
        self.conv = ConvNormRelu(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1 = torch.repeat_interleave(x1, 2, dim=2)
        # x1 = x1[:, :, :x2.shape[2]]
        x1 = torch.nn.functional.interpolate(x1, size=x2.shape[2], mode='linear')
        x = x1 + x2
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_dim, dim):
        super(UNet, self).__init__()
        # dim = 512
        self.down1 = nn.Sequential(
            ConvNormRelu(input_dim, input_dim, '1d', False),
            ConvNormRelu(input_dim, dim, '1d', False),
            ConvNormRelu(dim, dim, '1d', False)
        )
        self.gru = nn.GRU(dim, dim, 1, batch_first=True)
        self.down2 = ConvNormRelu(dim, dim, '1d', True)
        self.down3 = ConvNormRelu(dim, dim, '1d', True)
        self.down4 = ConvNormRelu(dim, dim, '1d', True)
        self.down5 = ConvNormRelu(dim, dim, '1d', True)
        self.down6 = ConvNormRelu(dim, dim, '1d', True)
        self.up1 = UnetUp(dim, dim)
        self.up2 = UnetUp(dim, dim)
        self.up3 = UnetUp(dim, dim)
        self.up4 = UnetUp(dim, dim)
        self.up5 = UnetUp(dim, dim)

    def forward(self, x1, pre_pose=None, w_pre=False):
        x2_0 = self.down1(x1)
        if w_pre:
            i = 1
            x2_pre = self.gru(x2_0[:,:,0:i].permute(0,2,1), pre_pose[:,:,-1:].permute(2,0,1).contiguous())[0].permute(0,2,1)
            x2 = torch.cat([x2_pre, x2_0[:,:,i:]], dim=-1)
            # x2 = torch.cat([pre_pose, x2_0], dim=2) # [B, 512, 15]
        else:
            # x2 = self.gru(x2_0.transpose(1, 2))[0].transpose(1,2)
            x2 = x2_0
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)             # [B, 512, 15]
        return x, x2_0


class AudioEncoder(nn.Module):
    def __init__(self, n_frames, template_length, pose=False, common_dim=512):
        super().__init__()
        self.n_frames = n_frames
        self.pose = pose
        self.step = 0
        self.weight = 0
        if self.pose:
            # self.first_net = nn.Sequential(
            #     ConvNormRelu(1, 64, '2d', False),
            #     ConvNormRelu(64, 64, '2d', True),
            #     ConvNormRelu(64, 128, '2d', False),
            #     ConvNormRelu(128, 128, '2d', True),
            #     ConvNormRelu(128, 256, '2d', False),
            #     ConvNormRelu(256, 256, '2d', True),
            #     ConvNormRelu(256, 256, '2d', False),
            #     ConvNormRelu(256, 256, '2d', False, padding='VALID')
            # )
            # decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4,
            #                                            dim_feedforward=2 * args.feature_dim, batch_first=True)
            # a = nn.TransformerDecoder
            self.first_net = SeqTranslator1D(256, 256,
                                             min_layers_num=4,
                                             residual=True
                                             )
            self.dropout_0 = nn.Dropout(0.1)
            self.mu_fc = nn.Conv1d(256, 128, 1, 1)
            self.var_fc = nn.Conv1d(256, 128, 1, 1)
            self.trans_motion = SeqTranslator1D(common_dim, common_dim,
                                                kernel_size=1,
                                                stride=1,
                                                min_layers_num=3,
                                                residual=True
                                                )
            # self.att = nn.MultiheadAttention(64 + template_length, 4, dropout=0.1)
            self.unet = UNet(128 + template_length, common_dim)

        else:
            self.first_net = SeqTranslator1D(256, 256,
                                             min_layers_num=4,
                                             residual=True
                                             )
            self.dropout_0 = nn.Dropout(0.1)
            # self.att = nn.MultiheadAttention(256, 4, dropout=0.1)
            self.unet = UNet(256, 256)
            self.dropout_1 = nn.Dropout(0.0)

    def forward(self, spectrogram, time_steps=None, template=None, pre_pose=None, w_pre=False):
        self.step = self.step + 1
        if self.pose:
            spect = spectrogram.transpose(1, 2)
            if w_pre:
                spect = spect[:, :, :]

            out = self.first_net(spect)
            out = self.dropout_0(out)

            mu = self.mu_fc(out)
            var = self.var_fc(out)
            audio = self.__reparam(mu, var)
            # audio = out

            # template = self.trans_motion(template)
            x1 = torch.cat([audio, template], dim=1)#.permute(2,0,1)
            # x1 = out
            #x1, _ = self.att(x1, x1, x1)
            #x1 = x1.permute(1,2,0)
            x1, x2_0 = self.unet(x1, pre_pose=pre_pose, w_pre=w_pre)
        else:
            spectrogram = spectrogram.transpose(1, 2)
            x1 = self.first_net(spectrogram)#.permute(2,0,1)
            #out, _ = self.att(out, out, out)
            #out = out.permute(1, 2, 0)
            x1 = self.dropout_0(x1)
            x1, x2_0 = self.unet(x1)
            x1 = self.dropout_1(x1)
            mu = None
            var = None

        return x1, (mu, var), x2_0

    def __reparam(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, device='cuda')
        z = eps * std + mu
        return z


class Generator(nn.Module):
    def __init__(self,
                 n_poses,
                 pose_dim,
                 pose,
                 n_pre_poses,
                 each_dim: list,
                 dim_list: list,
                 use_template=False,
                 template_length=0,
                 training=False,
                 device=None,
                 separate=False,
                 expression=False
                 ):
        super().__init__()

        self.use_template = use_template
        self.template_length = template_length
        self.training = training
        self.device = device
        self.separate = separate
        self.pose = pose
        self.decoderf = True
        self.expression = expression

        common_dim = 256

        if self.use_template:
            assert template_length > 0
            # self.KLLoss = KLLoss(kl_tolerance=self.config.Train.weights.kl_tolerance).to(self.device)
            # self.pose_encoder = SeqEncoder1D(
            #     C_in=pose_dim,
            #     C_out=512,
            #     T_in=n_poses,
            #     min_layer_nums=6
            #
            # )
            self.pose_encoder = SeqTranslator1D(pose_dim - 50, common_dim,
                                                # kernel_size=1,
                                                # stride=1,
                                                min_layers_num=3,
                                                residual=True
                                                )
            self.mu_fc = nn.Conv1d(common_dim, template_length, kernel_size=1, stride=1)
            self.var_fc = nn.Conv1d(common_dim, template_length, kernel_size=1, stride=1)

        else:
            self.template_length = 0

        self.gen_length = n_poses

        self.audio_encoder = AudioEncoder(n_poses, template_length, True, common_dim)
        self.speech_encoder = AudioEncoder(n_poses, template_length, False)

        # self.pre_pose_encoder = SeqEncoder1D(
        #     C_in=pose_dim,
        #     C_out=128,
        #     T_in=15,
        #     min_layer_nums=3
        #
        # )
        # self.pmu_fc = nn.Linear(128, 64)
        # self.pvar_fc = nn.Linear(128, 64)

        self.pre_pose_encoder = SeqTranslator1D(pose_dim-50, common_dim,
                                                min_layers_num=5,
                                                residual=True
                                                )
        self.decoder_in = 256 + 64
        self.dim_list = dim_list

        if self.separate:
            self.decoder = nn.ModuleList()
            self.final_out = nn.ModuleList()

            self.decoder.append(nn.Sequential(
                ConvNormRelu(256, 64),
                ConvNormRelu(64, 64),
                ConvNormRelu(64, 64),
            ))
            self.final_out.append(nn.Conv1d(64, each_dim[0], 1, 1))

            self.decoder.append(nn.Sequential(
                ConvNormRelu(common_dim, common_dim),
                ConvNormRelu(common_dim, common_dim),
                ConvNormRelu(common_dim, common_dim),
            ))
            self.final_out.append(nn.Conv1d(common_dim, each_dim[1], 1, 1))

            self.decoder.append(nn.Sequential(
                ConvNormRelu(common_dim, common_dim),
                ConvNormRelu(common_dim, common_dim),
                ConvNormRelu(common_dim, common_dim),
            ))
            self.final_out.append(nn.Conv1d(common_dim, each_dim[2], 1, 1))

            if self.expression:
                self.decoder.append(nn.Sequential(
                    ConvNormRelu(256, 256),
                    ConvNormRelu(256, 256),
                    ConvNormRelu(256, 256),
                ))
                self.final_out.append(nn.Conv1d(256, each_dim[3], 1, 1))
        else:
            self.decoder = nn.Sequential(
                ConvNormRelu(self.decoder_in, 512),
                ConvNormRelu(512, 512),
                ConvNormRelu(512, 512),
                ConvNormRelu(512, 512),
                ConvNormRelu(512, 512),
                ConvNormRelu(512, 512),
            )
            self.final_out = nn.Conv1d(512, pose_dim, 1, 1)

    def __reparam(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, device=self.device)
        z = eps * std + mu
        return z

    def forward(self, in_spec, pre_poses, gt_poses, template=None, time_steps=None, w_pre=False, norm=True):
        if time_steps is not None:
            self.gen_length = time_steps

        if self.use_template:
            if self.training:
                if w_pre:
                    in_spec = in_spec[:, 15:, :]
                    pre_pose = self.pre_pose_encoder(gt_poses[:, 14:15, :-50].permute(0, 2, 1))
                    pose_enc = self.pose_encoder(gt_poses[:, 15:, :-50].permute(0, 2, 1))
                    mu = self.mu_fc(pose_enc)
                    var = self.var_fc(pose_enc)
                    template = self.__reparam(mu, var)
                else:
                    pre_pose = None
                    pose_enc = self.pose_encoder(gt_poses[:, :, :-50].permute(0, 2, 1))
                    mu = self.mu_fc(pose_enc)
                    var = self.var_fc(pose_enc)
                    template = self.__reparam(mu, var)
            elif pre_poses is not None:
                if w_pre:
                    pre_pose = pre_poses[:, -1:, :-50]
                    if norm:
                        pre_pose = pre_pose.reshape(1, -1, 55, 5)
                        pre_pose = torch.cat([F.normalize(pre_pose[..., :3], dim=-1),
                                             F.normalize(pre_pose[..., 3:5], dim=-1)],
                                             dim=-1).reshape(1, -1, 275)
                    pre_pose = self.pre_pose_encoder(pre_pose.permute(0, 2, 1))
                    template = torch.randn([in_spec.shape[0], self.template_length, self.gen_length ]).to(
                        in_spec.device)
                else:
                    pre_pose = None
                    template = torch.randn([in_spec.shape[0], self.template_length, self.gen_length]).to(in_spec.device)
            elif gt_poses is not None:
                template = self.pre_pose_encoder(gt_poses[:, :, :-50].permute(0, 2, 1))
            elif template is None:
                pre_pose = None
                template = torch.randn([in_spec.shape[0], self.template_length, self.gen_length]).to(in_spec.device)
        else:
            template = None
            mu = None
            var = None

        a_t_f, (mu2, var2), x2_0 = self.audio_encoder(in_spec, time_steps=time_steps, template=template, pre_pose=pre_pose, w_pre=w_pre)
        s_f, _, _ = self.speech_encoder(in_spec, time_steps=time_steps)

        out = []

        if self.separate:
            for i in range(self.decoder.__len__()):
                if i == 0 or i == 3:
                    mid = self.decoder[i](s_f)
                else:
                    mid = self.decoder[i](a_t_f)
                mid = self.final_out[i](mid)
                out.append(mid)
            out = torch.cat(out, dim=1)

        else:
            out = self.decoder(a_t_f)
            out = self.final_out(out)

        out = out.transpose(1, 2)

        if self.training:
            if w_pre:
                return out, template, mu, var, (mu2, var2, x2_0, pre_pose)
            else:
                return out, template, mu, var, (mu2, var2, None, None)
        else:
            return out


class Discriminator(nn.Module):
    def __init__(self, pose_dim, pose):
        super().__init__()
        self.net = nn.Sequential(
            Conv1d_tf(pose_dim, 64, kernel_size=4, stride=2, padding='SAME'),
            nn.LeakyReLU(0.2, True),
            ConvNormRelu(64, 128, '1d', True),
            ConvNormRelu(128, 256, '1d', k=4, s=1),
            Conv1d_tf(256, 1, kernel_size=4, stride=1, padding='SAME'),
        )

    def forward(self, x):
        x = x.transpose(1, 2)

        out = self.net(x)
        return out


def main():
    d = Discriminator(275, 55)
    x = torch.randn([8, 60, 275])
    result = d(x)


if __name__ == "__main__":
    main()
