import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt




class CasualCT(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 leaky=False,
                 p=0,
                 groups=1, ):
        '''
        conv-bn-relu
        '''
        super(CasualCT, self).__init__()
        padding = 0
        kernel_size = 2
        stride = 2
        in_channels = in_channels * groups
        out_channels = out_channels * groups

        self.conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding,
                                       groups=groups)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=p)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        out = self.norm(self.dropout(self.conv(x)))
        return self.relu(out)


class CasualConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 leaky=False,
                 p=0,
                 groups=1,
                 downsample=False):
        '''
        conv-bn-relu
        '''
        super(CasualConv, self).__init__()
        padding = 0
        kernel_size = 2
        stride = 1
        self.downsample = downsample
        if self.downsample:
            kernel_size = 2
            stride = 2

        in_channels = in_channels * groups
        out_channels = out_channels * groups
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=groups)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=p)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU()

    def forward(self, x, pre_state=None):
        if not self.downsample:
            if pre_state is not None:
                x = torch.cat([pre_state, x], dim=-1)
            else:
                zeros = torch.zeros([x.shape[0], x.shape[1], 1], device=x.device)
                x = torch.cat([zeros, x], dim=-1)
        out = self.norm(self.dropout(self.conv(x)))
        return self.relu(out)


class ConvNormRelu(nn.Module):
    '''
    (B,C_in,H,W) -> (B, C_out, H, W)
    there exist some kernel size that makes the result is not H/s
    #TODO: there might some problems with residual
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 leaky=False,
                 sample='none',
                 p=0,
                 groups=1,
                 residual=False,
                 norm='bn'):
        '''
        conv-bn-relu
        '''
        super(ConvNormRelu, self).__init__()
        self.residual = residual
        self.norm_type = norm
        padding = 1

        if sample == 'none':
            kernel_size = 3
            stride = 1
        elif sample == 'one':
            padding = 0
            kernel_size = stride = 1
        else:
            kernel_size = 4
            stride = 2

        if self.residual:
            if sample == 'down':
                self.residual_layer = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)
            elif sample == 'up':
                self.residual_layer = nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)
            else:
                if in_channels == out_channels:
                    self.residual_layer = nn.Identity()
                else:
                    self.residual_layer = nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                        )
                    )

        in_channels = in_channels * groups
        out_channels = out_channels * groups
        if sample == 'up':
            self.conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, stride=stride, padding=padding,
                                           groups=groups)
        else:
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  groups=groups)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=p)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        out = self.norm(self.dropout(self.conv(x)))
        if self.residual:
            residual = self.residual_layer(x)
            out += residual
        return self.relu(out)


class Res_CNR_Stack(nn.Module):
    def __init__(self,
                 channels,
                 layers,
                 sample='none',
                 leaky=False,
                 casual=False,
                 ):
        super(Res_CNR_Stack, self).__init__()

        if casual:
            kernal_size = 1
            padding = 0
            conv = CasualConv
        else:
            kernal_size = 3
            padding = 1
            conv = ConvNormRelu

        if sample == 'one':
            kernal_size = 1
            padding = 0

        self._layers = nn.ModuleList()
        for i in range(layers):
            self._layers.append(conv(channels, channels, leaky=leaky, sample=sample))
        self.conv = nn.Conv1d(channels, channels, kernal_size, 1, padding)
        self.norm = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x, pre_state=None):
        # cur_state = []
        h = x
        for i in range(self._layers.__len__()):
            # cur_state.append(h[..., -1:])
            h = self._layers[i](h, pre_state=pre_state[i] if pre_state is not None else None)
        h = self.norm(self.conv(h))
        return self.relu(h + x)


class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.

      This module keeps track of a hidden exponential moving average that is
      initialized as a vector of zeros which is then normalized to give the average.
      This gives us a moving average which isn't biased towards either zero or the
      initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

      Initially:
          hidden_0 = 0
      Then iteratively:
          hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
          average_i = hidden_i / (1 - decay^i)
    """

    def __init__(self, init_value, decay):
        super().__init__()

        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))

    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay ** self.counter)
        return average


class VectorQuantizerEMA(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. Use EMA to update embeddings.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
        decay (float): decay for the moving averages.
        epsilon (float): small float constant to avoid numerical instability.
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay,
                 epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon

        # initialize embeddings as buffers
        embeddings = torch.empty(self.num_embeddings, self.embedding_dim)
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)

        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((self.num_embeddings,)), decay)

    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 1).contiguous()
        # [B, H, W, C] -> [BHW, C]
        flat_x = x.reshape(-1, self.embedding_dim)

        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)  # [B, W, C]

        if not self.training:
            quantized = quantized.permute(0, 2, 1).contiguous()
            return quantized, encoding_indices.view(quantized.shape[0], quantized.shape[2])

        # update embeddings with EMA
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            updated_ema_cluster_size = self.ema_cluster_size(torch.sum(encodings, dim=0))
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                        (n + self.num_embeddings * self.epsilon) * n)
            dw = torch.matmul(encodings.t(), flat_x)  # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
                    updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1))
            self.embeddings.data = normalised_updated_ema_w

        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()

        quantized = quantized.permute(0, 2, 1).contiguous()
        return quantized, loss

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(self.embeddings ** 2, dim=1) -
                2. * torch.matmul(flat_x, self.embeddings.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)



class Casual_Encoder(nn.Module):
    def __init__(self, in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Casual_Encoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.project = nn.Conv1d(in_dim, self._num_hiddens // 4, 1, 1)
        self._enc_1 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True, casual=True)
        self._down_1 = CasualConv(self._num_hiddens // 4, self._num_hiddens // 2, leaky=True, downsample=True)
        self._enc_2 = Res_CNR_Stack(self._num_hiddens // 2, self._num_residual_layers, leaky=True, casual=True)
        self._down_2 = CasualConv(self._num_hiddens // 2, self._num_hiddens, leaky=True, downsample=True)
        self._enc_3 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True, casual=True)
        # self.pre_vq_conv = nn.Conv1d(self._num_hiddens, embedding_dim, 1, 1)

    def forward(self, x):
        h = self.project(x)
        h, _ = self._enc_1(h)
        h = self._down_1(h)
        h, _ = self._enc_2(h)
        h = self._down_2(h)
        h, _ = self._enc_3(h)
        # h = self.pre_vq_conv(h)
        return h


class Casual_Decoder(nn.Module):
    def __init__(self, out_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Casual_Decoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        # self.aft_vq_conv = nn.Conv1d(embedding_dim, self._num_hiddens, 1, 1)
        self._dec_1 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True, casual=True)
        self._up_2 = CasualCT(self._num_hiddens, self._num_hiddens // 2, leaky=True)
        self._dec_2 = Res_CNR_Stack(self._num_hiddens // 2, self._num_residual_layers, leaky=True, casual=True)
        self._up_3 = CasualCT(self._num_hiddens // 2, self._num_hiddens // 4, leaky=True)
        self._dec_3 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True, casual=True)
        self.project = nn.Conv1d(self._num_hiddens//4, out_dim, 1, 1)

    def forward(self, h, pre_state=None):
        cur_state = []
        # h = self.aft_vq_conv(x)
        h, s = self._dec_1(h, pre_state[0] if pre_state is not None else None)
        cur_state.append(s)
        h = self._up_2(h)
        h, s = self._dec_2(h, pre_state[1] if pre_state is not None else None)
        cur_state.append(s)
        h = self._up_3(h)
        h, s = self._dec_3(h, pre_state[2] if pre_state is not None else None)
        cur_state.append(s)
        recon = self.project(h)
        return recon, cur_state