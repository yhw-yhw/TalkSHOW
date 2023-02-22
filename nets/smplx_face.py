import os
import sys

sys.path.append(os.getcwd())

from nets.layers import *
from nets.base import TrainWrapperBaseClass
# from nets.spg.faceformer import Faceformer
from nets.spg.s2g_face import Generator as s2g_face
from losses import KeypointLoss
from nets.utils import denormalize
from data_utils import get_mfcc_psf, get_mfcc_psf_min, get_mfcc_ta
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import smplx


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

        self.generator = s2g_face(
            n_poses=self.config.Data.pose.generate_length,
            each_dim=self.each_dim,
            dim_list=self.dim_list,
            training=not self.args.infer,
            device=self.device,
            identity=False if self.convert_to_6d else True,
            num_classes=self.num_classes,
        ).to(self.device)

        # self.generator = Faceformer().to(self.device)

        self.discriminator = None
        self.am = None

        self.MSELoss = KeypointLoss().to(self.device)
        super().__init__(args, config)

    def init_optimizer(self):
        self.generator_optimizer = optim.SGD(
            filter(lambda p: p.requires_grad,self.generator.parameters()),
            lr=0.001,
            momentum=0.9,
            nesterov=False,
        )

    def init_params(self):
        if self.convert_to_6d:
            scale = 2
        else:
            scale = 1

        global_orient = round(3 * scale)
        leye_pose = reye_pose = round(3 * scale)
        jaw_pose = round(3 * scale)
        body_pose = round(63 * scale)
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
        self.full_dim = jaw_dim + eye_dim + body_dim + hand_dim + face_dim
        self.pose = int(self.full_dim / round(3 * scale))
        self.each_dim = [jaw_dim, eye_dim + body_dim, hand_dim, face_dim]

    def __call__(self, bat):
        # assert (not self.args.infer), "infer mode"
        self.global_step += 1

        total_loss = None
        loss_dict = {}

        aud, poses = bat['aud_feat'].to(self.device).to(torch.float32), bat['poses'].to(self.device).to(torch.float32)
        id = bat['speaker'].to(self.device) - 20
        id = F.one_hot(id, self.num_classes)

        aud = aud.permute(0, 2, 1)
        gt_poses = poses.permute(0, 2, 1)

        if self.expression:
            expression = bat['expression'].to(self.device).to(torch.float32)
            gt_poses = torch.cat([gt_poses, expression.permute(0, 2, 1)], dim=2)

        pred_poses, _ = self.generator(
            aud,
            gt_poses,
            id,
        )

        G_loss, G_loss_dict = self.get_loss(
            pred_poses=pred_poses,
            gt_poses=gt_poses,
            pre_poses=None,
            mode='training_G',
            gt_conf=None,
            aud=aud,
        )

        self.generator_optimizer.zero_grad()
        G_loss.backward()
        grad = torch.nn.utils.clip_grad_norm(self.generator.parameters(), self.config.Train.max_gradient_norm)
        loss_dict['grad'] = grad.item()
        self.generator_optimizer.step()

        for key in list(G_loss_dict.keys()):
            loss_dict[key] = G_loss_dict.get(key, 0).item()

        return total_loss, loss_dict

    def get_loss(self,
                 pred_poses,
                 gt_poses,
                 pre_poses,
                 aud,
                 mode='training_G',
                 gt_conf=None,
                 exp=1,
                 gt_nzero=None,
                 pre_nzero=None,
                 ):
        loss_dict = {}


        [b_j, b_e, b_b, b_h, b_f] = self.dim_list

        MSELoss = torch.mean(torch.abs(pred_poses[:, :, :6] - gt_poses[:, :, :6]))
        if self.expression:
            expl = torch.mean((pred_poses[:, :, -100:] - gt_poses[:, :, -100:])**2)
        else:
            expl = 0

        gen_loss = expl + MSELoss

        loss_dict['MSELoss'] = MSELoss
        if self.expression:
            loss_dict['exp_loss'] = expl

        return gen_loss, loss_dict

    def infer_on_audio(self, aud_fn, id=None, initial_pose=None, norm_stats=None, w_pre=False, frame=None, am=None, am_sr=16000, **kwargs):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        # assert self.args.infer, "train mode"
        self.generator.eval()

        if self.config.Data.pose.normalization:
            assert norm_stats is not None
            data_mean = norm_stats[0]
            data_std = norm_stats[1]

        # assert initial_pose.shape[-1] == pre_length
        if initial_pose is not None:
            gt = initial_pose[:,:,:].permute(0, 2, 1).to(self.generator.device).to(torch.float32)
            pre_poses = initial_pose[:,:,:15].permute(0, 2, 1).to(self.generator.device).to(torch.float32)
            poses = initial_pose.permute(0, 2, 1).to(self.generator.device).to(torch.float32)
            B = pre_poses.shape[0]
        else:
            gt = None
            pre_poses=None
            B = 1

        if type(aud_fn) == torch.Tensor:
            aud_feat = torch.tensor(aud_fn, dtype=torch.float32).to(self.generator.device)
            num_poses_to_generate = aud_feat.shape[-1]
        else:
            aud_feat = get_mfcc_ta(aud_fn, am=am, am_sr=am_sr, fps=30, encoder_choice='faceformer')
            aud_feat = aud_feat[np.newaxis, ...].repeat(B, axis=0)
            aud_feat = torch.tensor(aud_feat, dtype=torch.float32).to(self.generator.device).transpose(1, 2)
        if frame is None:
            frame = aud_feat.shape[2]*30//16000
        #
        if id is None:
            id = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32, device=self.generator.device)
        else:
            id = F.one_hot(id, self.num_classes).to(self.generator.device)

        with torch.no_grad():
            pred_poses = self.generator(aud_feat, pre_poses, id, time_steps=frame)[0]
            pred_poses = pred_poses.cpu().numpy()
        output = pred_poses

        if self.config.Data.pose.normalization:
            output = denormalize(output, data_mean, data_std)

        return output


    def generate(self, wv2_feat, frame):
        '''
        initial_pose: (B, C, T), normalized
        (aud_fn, txgfile) -> generated motion (B, T, C)
        '''
        output = []

        # assert self.args.infer, "train mode"
        self.generator.eval()

        B = 1

        id = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32, device=self.generator.device)
        id = id.repeat(wv2_feat.shape[0], 1)

        with torch.no_grad():
            pred_poses = self.generator(wv2_feat, None, id, time_steps=frame)[0]
        return pred_poses
