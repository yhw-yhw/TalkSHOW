'''
not exactly the same as the official repo but the results are good
'''
import sys
import os

from transformers import Wav2Vec2Processor

from .wav2vec import Wav2Vec2Model
from torchaudio.sox_effects import apply_effects_tensor

sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
import math
from nets.layers import SeqEncoder1D, SeqTranslator1D, ConvNormRelu


""" from https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context.git """


def audio_chunking(audio: torch.Tensor, frame_rate: int = 30, chunk_size: int = 16000):
    """
    :param audio: 1 x T tensor containing a 16kHz audio signal
    :param frame_rate: frame rate for video (we need one audio chunk per video frame)
    :param chunk_size: number of audio samples per chunk
    :return: num_chunks x chunk_size tensor containing sliced audio
    """
    samples_per_frame = 16000 // frame_rate
    padding = (chunk_size - samples_per_frame) // 2
    audio = torch.nn.functional.pad(audio.unsqueeze(0), pad=[padding, padding]).squeeze(0)
    anchor_points = list(range(chunk_size//2, audio.shape[-1]-chunk_size//2, samples_per_frame))
    audio = torch.cat([audio[:, i-chunk_size//2:i+chunk_size//2] for i in anchor_points], dim=0)
    return audio


class MeshtalkEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, model_name: str = 'audio_encoder'):
        """
        :param latent_dim: size of the latent audio embedding
        :param model_name: name of the model, used to load and save the model
        """
        super().__init__()

        self.melspec = ta.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=2048, win_length=800, hop_length=160, n_mels=80
        )

        conv_len = 5
        self.convert_dimensions = torch.nn.Conv1d(80, 128, kernel_size=conv_len)
        self.weights_init(self.convert_dimensions)
        self.receptive_field = conv_len

        convs = []
        for i in range(6):
            dilation = 2 * (i % 3 + 1)
            self.receptive_field += (conv_len - 1) * dilation
            convs += [torch.nn.Conv1d(128, 128, kernel_size=conv_len, dilation=dilation)]
            self.weights_init(convs[-1])
        self.convs = torch.nn.ModuleList(convs)
        self.code = torch.nn.Linear(128, latent_dim)

        self.apply(lambda x: self.weights_init(x))

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            try:
                torch.nn.init.constant_(m.bias, .01)
            except:
                pass

    def forward(self, audio: torch.Tensor):
        """
        :param audio: B x T x 16000 Tensor containing 1 sec of audio centered around the current time frame
        :return: code: B x T x latent_dim Tensor containing a latent audio code/embedding
        """
        B, T = audio.shape[0], audio.shape[1]
        x = self.melspec(audio).squeeze(1)
        x = torch.log(x.clamp(min=1e-10, max=None))
        if T == 1:
            x = x.unsqueeze(1)

        # Convert to the right dimensionality
        x = x.view(-1, x.shape[2], x.shape[3])
        x = F.leaky_relu(self.convert_dimensions(x), .2)

        # Process stacks
        for conv in self.convs:
            x_ = F.leaky_relu(conv(x), .2)
            if self.training:
                x_ = F.dropout(x_, .2)
            l = (x.shape[2] - x_.shape[2]) // 2
            x = (x[:, :, l:-l] + x_) / 2

        x = torch.mean(x, dim=-1)
        x = x.view(B, T, x.shape[-1])
        x = self.code(x)

        return {"code": x}


class AudioEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, identity=False, num_classes=0):
        super().__init__()
        self.identity = identity
        if self.identity:
            in_dim = in_dim + 64
            self.id_mlp = nn.Conv1d(num_classes, 64, 1, 1)
        self.first_net = SeqTranslator1D(in_dim, out_dim,
                                         min_layers_num=3,
                                         residual=True,
                                         norm='ln'
                                         )
        self.grus = nn.GRU(out_dim, out_dim, 1, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        # self.att = nn.MultiheadAttention(out_dim, 4, dropout=0.1, batch_first=True)

    def forward(self, spectrogram, pre_state=None, id=None, time_steps=None):

        spectrogram = spectrogram
        spectrogram = self.dropout(spectrogram)
        if self.identity:
            id = id.reshape(id.shape[0], -1, 1).repeat(1, 1, spectrogram.shape[2]).to(torch.float32)
            id = self.id_mlp(id)
            spectrogram = torch.cat([spectrogram, id], dim=1)
        x1 = self.first_net(spectrogram)# .permute(0, 2, 1)
        if time_steps is not None:
            x1 = F.interpolate(x1, size=time_steps, align_corners=False, mode='linear')
        # x1, _ = self.att(x1, x1, x1)
        # x1, hidden_state = self.grus(x1)
        # x1 = x1.permute(0, 2, 1)
        hidden_state=None

        return x1, hidden_state


class Generator(nn.Module):
    def __init__(self,
                 n_poses,
                 each_dim: list,
                 dim_list: list,
                 training=False,
                 device=None,
                 identity=True,
                 num_classes=0,
                 ):
        super().__init__()

        self.training = training
        self.device = device
        self.gen_length = n_poses
        self.identity = identity

        norm = 'ln'
        in_dim = 256
        out_dim = 256

        self.encoder_choice = 'faceformer'

        if self.encoder_choice == 'meshtalk':
            self.audio_encoder = MeshtalkEncoder(latent_dim=in_dim)
        elif self.encoder_choice == 'faceformer':
            # wav2vec 2.0 weights initialization
            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")  # "vitouphy/wav2vec2-xls-r-300m-phoneme""facebook/wav2vec2-base-960h"
            self.audio_encoder.feature_extractor._freeze_parameters()
            self.audio_feature_map = nn.Linear(768, in_dim)
        else:
            self.audio_encoder = AudioEncoder(in_dim=64, out_dim=out_dim)

        self.audio_middle = AudioEncoder(in_dim, out_dim, identity, num_classes)

        self.dim_list = dim_list

        self.decoder = nn.ModuleList()
        self.final_out = nn.ModuleList()

        self.decoder.append(nn.Sequential(
            ConvNormRelu(out_dim, 64, norm=norm),
            ConvNormRelu(64, 64, norm=norm),
            ConvNormRelu(64, 64, norm=norm),
        ))
        self.final_out.append(nn.Conv1d(64, each_dim[0], 1, 1))

        self.decoder.append(nn.Sequential(
            ConvNormRelu(out_dim, out_dim, norm=norm),
            ConvNormRelu(out_dim, out_dim, norm=norm),
            ConvNormRelu(out_dim, out_dim, norm=norm),
        ))
        self.final_out.append(nn.Conv1d(out_dim, each_dim[3], 1, 1))

    def forward(self, in_spec, gt_poses=None, id=None, pre_state=None, time_steps=None):
        if self.training:
            time_steps = gt_poses.shape[1]

        # vector, hidden_state = self.audio_encoder(in_spec, pre_state, time_steps=time_steps)
        if self.encoder_choice == 'meshtalk':
            in_spec = audio_chunking(in_spec.squeeze(-1), frame_rate=30, chunk_size=16000)
            feature = self.audio_encoder(in_spec.unsqueeze(0))["code"].transpose(1, 2)
        elif self.encoder_choice == 'faceformer':
            hidden_states = self.audio_encoder(in_spec.reshape(in_spec.shape[0], -1), frame_num=time_steps).last_hidden_state
            feature = self.audio_feature_map(hidden_states).transpose(1, 2)
        else:
            feature, hidden_state = self.audio_encoder(in_spec, pre_state, time_steps=time_steps)

        # hidden_states = in_spec

        feature, _ = self.audio_middle(feature, id=id)

        out = []

        for i in range(self.decoder.__len__()):
            mid = self.decoder[i](feature)
            mid = self.final_out[i](mid)
            out.append(mid)

        out = torch.cat(out, dim=1)
        out = out.transpose(1, 2)

        return out, None


