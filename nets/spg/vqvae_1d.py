import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .wav2vec import Wav2Vec2Model
from .vqvae_modules import VectorQuantizerEMA, ConvNormRelu, Res_CNR_Stack



class AudioEncoder(nn.Module):
    def __init__(self, in_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(AudioEncoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.project = ConvNormRelu(in_dim, self._num_hiddens // 4, leaky=True)

        self._enc_1 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True)
        self._down_1 = ConvNormRelu(self._num_hiddens // 4, self._num_hiddens // 2, leaky=True, residual=True,
                                    sample='down')
        self._enc_2 = Res_CNR_Stack(self._num_hiddens // 2, self._num_residual_layers, leaky=True)
        self._down_2 = ConvNormRelu(self._num_hiddens // 2, self._num_hiddens, leaky=True, residual=True, sample='down')
        self._enc_3 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)

    def forward(self, x, frame_num=0):
        h = self.project(x)
        h = self._enc_1(h)
        h = self._down_1(h)
        h = self._enc_2(h)
        h = self._down_2(h)
        h = self._enc_3(h)
        return h


class Wav2VecEncoder(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers):
        super(Wav2VecEncoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers

        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h")  # "vitouphy/wav2vec2-xls-r-300m-phoneme""facebook/wav2vec2-base-960h"
        self.audio_encoder.feature_extractor._freeze_parameters()

        self.project = ConvNormRelu(768, self._num_hiddens, leaky=True)

        self._enc_1 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)
        self._down_1 = ConvNormRelu(self._num_hiddens, self._num_hiddens, leaky=True, residual=True, sample='down')
        self._enc_2 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)
        self._down_2 = ConvNormRelu(self._num_hiddens, self._num_hiddens, leaky=True, residual=True, sample='down')
        self._enc_3 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)

    def forward(self, x, frame_num):
        h = self.audio_encoder(x.squeeze(), frame_num=frame_num).last_hidden_state.transpose(1, 2)
        h = self.project(h)
        h = self._enc_1(h)
        h = self._down_1(h)
        h = self._enc_2(h)
        h = self._down_2(h)
        h = self._enc_3(h)
        return h


class Encoder(nn.Module):
    def __init__(self, in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.project = ConvNormRelu(in_dim, self._num_hiddens // 4, leaky=True)

        self._enc_1 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True)
        self._down_1 = ConvNormRelu(self._num_hiddens // 4, self._num_hiddens // 2, leaky=True, residual=True,
                                    sample='down')
        self._enc_2 = Res_CNR_Stack(self._num_hiddens // 2, self._num_residual_layers, leaky=True)
        self._down_2 = ConvNormRelu(self._num_hiddens // 2, self._num_hiddens, leaky=True, residual=True, sample='down')
        self._enc_3 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)

        self.pre_vq_conv = nn.Conv1d(self._num_hiddens, embedding_dim, 1, 1)

    def forward(self, x):
        h = self.project(x)
        h = self._enc_1(h)
        h = self._down_1(h)
        h = self._enc_2(h)
        h = self._down_2(h)
        h = self._enc_3(h)
        h = self.pre_vq_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(self, out_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.aft_vq_conv = nn.Conv1d(embedding_dim, self._num_hiddens, 1, 1)

        self._dec_1 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)
        self._up_2 = ConvNormRelu(self._num_hiddens, self._num_hiddens // 2, leaky=True, residual=True, sample='up')
        self._dec_2 = Res_CNR_Stack(self._num_hiddens // 2, self._num_residual_layers, leaky=True)
        self._up_3 = ConvNormRelu(self._num_hiddens // 2, self._num_hiddens // 4, leaky=True, residual=True,
                                  sample='up')
        self._dec_3 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True)

        self.project = nn.Conv1d(self._num_hiddens // 4, out_dim, 1, 1)

    def forward(self, h, last_frame=None):

        h = self.aft_vq_conv(h)
        h = self._dec_1(h)
        h = self._up_2(h)
        h = self._dec_2(h)
        h = self._up_3(h)
        h = self._dec_3(h)

        recon = self.project(h)
        return recon, None


class Pre_VQ(nn.Module):
    def __init__(self, num_hiddens, embedding_dim, num_chunks):
        super(Pre_VQ, self).__init__()
        self.conv = nn.Conv1d(num_hiddens, num_hiddens, 1, 1, 0, groups=num_chunks)
        self.bn = nn.GroupNorm(num_chunks, num_hiddens)
        self.relu = nn.ReLU()
        self.proj = nn.Conv1d(num_hiddens, embedding_dim, 1, 1, 0, groups=num_chunks)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.proj(x)
        return x


class VQVAE(nn.Module):
    """VQ-VAE"""

    def __init__(self, in_dim, embedding_dim, num_embeddings,
                 num_hiddens, num_residual_layers, num_residual_hiddens,
                 commitment_cost=0.25, decay=0.99, share=False):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.share_code_vq = share

        self.encoder = Encoder(in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.vq_layer = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.decoder = Decoder(in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, gt_poses, id=None, pre_state=None):
        z = self.encoder(gt_poses.transpose(1, 2))
        if not self.training:
            e, _ = self.vq_layer(z)
            x_recon, cur_state = self.decoder(e, pre_state.transpose(1, 2) if pre_state is not None else None)
            return e, x_recon

        e, e_q_loss = self.vq_layer(z)
        gt_recon, cur_state = self.decoder(e, pre_state.transpose(1, 2) if pre_state is not None else None)

        return e_q_loss, gt_recon.transpose(1, 2)

    def encode(self, gt_poses, id=None):
        z = self.encoder(gt_poses.transpose(1, 2))
        e, latents = self.vq_layer(z)
        return e, latents

    def decode(self, b, w, e=None, latents=None, pre_state=None):
        if e is not None:
            x = self.decoder(e, pre_state.transpose(1, 2) if pre_state is not None else None)
        else:
            e = self.vq_layer.quantize(latents)
            e = e.view(b, w, -1).permute(0, 2, 1).contiguous()
            x = self.decoder(e, pre_state.transpose(1, 2) if pre_state is not None else None)
        return x

