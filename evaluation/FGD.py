import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
import math
from data_utils.rotation_conversion import axis_angle_to_matrix, matrix_to_rotation_6d

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore warnings


change_angle = torch.tensor([6.0181e-05, 5.1597e-05, 2.1344e-04, 2.1899e-04])
class EmbeddingSpaceEvaluator:
    def __init__(self, ae, vae, device):

        # init embed net
        self.ae = ae
        # self.vae = vae

        # storage
        self.real_feat_list = []
        self.generated_feat_list = []
        self.real_joints_list = []
        self.generated_joints_list = []
        self.real_6d_list = []
        self.generated_6d_list = []
        self.audio_beat_list = []

    def reset(self):
        self.real_feat_list = []
        self.generated_feat_list = []

    def get_no_of_samples(self):
        return len(self.real_feat_list)

    def push_samples(self, generated_poses, real_poses):
        # self.net.eval()
        # convert poses to latent features
        real_feat, real_poses = self.ae.extract(real_poses)
        generated_feat, generated_poses = self.ae.extract(generated_poses)

        num_joints = real_poses.shape[2] // 3

        real_feat = real_feat.squeeze()
        generated_feat = generated_feat.reshape(generated_feat.shape[0]*generated_feat.shape[1], -1)

        self.real_feat_list.append(real_feat.data.cpu().numpy())
        self.generated_feat_list.append(generated_feat.data.cpu().numpy())

        # real_poses = matrix_to_rotation_6d(axis_angle_to_matrix(real_poses.reshape(-1, 3))).reshape(-1, num_joints, 6)
        # generated_poses = matrix_to_rotation_6d(axis_angle_to_matrix(generated_poses.reshape(-1, 3))).reshape(-1, num_joints, 6)
        #
        # self.real_feat_list.append(real_poses.data.cpu().numpy())
        # self.generated_feat_list.append(generated_poses.data.cpu().numpy())

    def push_joints(self, generated_poses, real_poses):
        self.real_joints_list.append(real_poses.data.cpu())
        self.generated_joints_list.append(generated_poses.squeeze().data.cpu())

    def push_aud(self, aud):
        self.audio_beat_list.append(aud.squeeze().data.cpu())

    def get_MAAC(self):
        ang_vel_list = []
        for real_joints in self.real_joints_list:
            real_joints[:, 15:21] = real_joints[:, 16:22]
            vec = real_joints[:, 15:21] - real_joints[:, 13:19]
            inner_product = torch.einsum('kij,kij->ki', [vec[:, 2:], vec[:, :-2]])
            inner_product = torch.clamp(inner_product, -1, 1, out=None)
            angle = torch.acos(inner_product) / math.pi
            ang_vel = (angle[1:] - angle[:-1]).abs().mean(dim=0)
            ang_vel_list.append(ang_vel.unsqueeze(dim=0))
        all_vel = torch.cat(ang_vel_list, dim=0)
        MAAC = all_vel.mean(dim=0)
        return MAAC

    def get_BCscore(self):
        thres = 0.01
        sigma = 0.1
        sum_1 = 0
        total_beat = 0
        for joints, audio_beat_time in zip(self.generated_joints_list, self.audio_beat_list):
            motion_beat_time = []
            if joints.dim() == 4:
                joints = joints[0]
            joints[:, 15:21] = joints[:, 16:22]
            vec = joints[:, 15:21] - joints[:, 13:19]
            inner_product = torch.einsum('kij,kij->ki', [vec[:, 2:], vec[:, :-2]])
            inner_product = torch.clamp(inner_product, -1, 1, out=None)
            angle = torch.acos(inner_product) / math.pi
            ang_vel = (angle[1:] - angle[:-1]).abs() / change_angle / len(change_angle)

            angle_diff = torch.cat((torch.zeros(1, 4), ang_vel), dim=0)

            sum_2 = 0
            for i in range(angle_diff.shape[1]):
                motion_beat_time = []
                for t in range(1, joints.shape[0]-1):
                    if (angle_diff[t][i] < angle_diff[t - 1][i] and angle_diff[t][i] < angle_diff[t + 1][i]):
                        if (angle_diff[t - 1][i] - angle_diff[t][i] >= thres or angle_diff[t + 1][i] - angle_diff[
                            t][i] >= thres):
                            motion_beat_time.append(float(t) / 30.0)
                if (len(motion_beat_time) == 0):
                    continue
                motion_beat_time = torch.tensor(motion_beat_time)
                sum = 0
                for audio in audio_beat_time:
                    sum += np.power(math.e, -(np.power((audio.item() - motion_beat_time), 2)).min() / (2 * sigma * sigma))
                sum_2 = sum_2 + sum
                total_beat = total_beat + len(audio_beat_time)
            sum_1 = sum_1 + sum_2
        return sum_1/total_beat


    def get_scores(self):
        generated_feats = np.vstack(self.generated_feat_list)
        real_feats = np.vstack(self.real_feat_list)

        def frechet_distance(samples_A, samples_B):
            A_mu = np.mean(samples_A, axis=0)
            A_sigma = np.cov(samples_A, rowvar=False)
            B_mu = np.mean(samples_B, axis=0)
            B_sigma = np.cov(samples_B, rowvar=False)
            try:
                frechet_dist = self.calculate_frechet_distance(A_mu, A_sigma, B_mu, B_sigma)
            except ValueError:
                frechet_dist = 1e+10
            return frechet_dist

        ####################################################################
        # frechet distance
        frechet_dist = frechet_distance(generated_feats, real_feats)

        ####################################################################
        # distance between real and generated samples on the latent feature space
        dists = []
        for i in range(real_feats.shape[0]):
            d = np.sum(np.absolute(real_feats[i] - generated_feats[i]))  # MAE
            dists.append(d)
        feat_dist = np.mean(dists)

        return frechet_dist, feat_dist

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """ from https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)