'''
LVD: different initial pose
diversity: same initial pose
'''
import os
import sys
sys.path.append(os.getcwd())

from glob import glob

from argparse import ArgumentParser
import json

from evaluation.util import *
from evaluation.metrics import *
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--speaker', required=True, type=str)
parser.add_argument('--post_fix', nargs='+', default=['base'], type=str)
args = parser.parse_args()

speaker = args.speaker
test_audios = sorted(glob('pose_dataset/videos/test_audios/%s/*.wav'%(speaker)))

LVD_list = []
diversity_list = []

for aud in tqdm(test_audios):
    base_name = os.path.splitext(aud)[0]
    gt_path = get_full_path(aud, speaker, 'val')
    _, gt_poses, _ = get_gts(gt_path)
    gt_poses = gt_poses[np.newaxis,...]
    # print(gt_poses.shape)#(seq_len, 135*2)pose, lhand, rhand, face
    for post_fix in args.post_fix:
        pred_path = base_name + '_'+post_fix+'.json'
        pred_poses = np.array(json.load(open(pred_path)))
        # print(pred_poses.shape)#(B, seq_len, 108)
        pred_poses = cvt25(pred_poses, gt_poses)
        # print(pred_poses.shape)#(B, seq, pose_dim)

        gt_valid_points = hand_points(gt_poses)
        pred_valid_points = hand_points(pred_poses)

        lvd = LVD(gt_valid_points, pred_valid_points)
        # div = diversity(pred_valid_points)

        LVD_list.append(lvd)
        # diversity_list.append(div)

        # gt_velocity = peak_velocity(gt_valid_points, order=2)
        # pred_velocity = peak_velocity(pred_valid_points, order=2)

        # gt_consistency = velocity_consistency(gt_velocity, pred_velocity)
        # pred_consistency = velocity_consistency(pred_velocity, gt_velocity)

        # gt_consistency_list.append(gt_consistency)
        # pred_consistency_list.append(pred_consistency)

lvd = np.mean(LVD_list)
# diversity_list = np.mean(diversity_list)

print('LVD:', lvd)
# print("diversity:", diversity_list)