'''
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
parser.add_argument('--post_fix', nargs='+', default=['paper_model'], type=str)
args = parser.parse_args()

speaker = args.speaker
test_audios = sorted(glob('pose_dataset/videos/test_audios/%s/*.wav'%(speaker)))

quality_samples={'gt':[]}
for post_fix in args.post_fix:
    quality_samples[post_fix] = []

for aud in tqdm(test_audios):
    base_name = os.path.splitext(aud)[0]
    gt_path = get_full_path(aud, speaker, 'val')
    _, gt_poses, _ = get_gts(gt_path)
    gt_poses = gt_poses[np.newaxis,...]
    gt_valid_points = valid_points(gt_poses)
    # print(gt_valid_points.shape)
    quality_samples['gt'].append(gt_valid_points)

    for post_fix in args.post_fix:
        pred_path = base_name + '_'+post_fix+'.json'
        pred_poses = np.array(json.load(open(pred_path)))
        # print(pred_poses.shape)#(B, seq_len, 108)
        pred_poses = cvt25(pred_poses, gt_poses)
        # print(pred_poses.shape)#(B, seq, pose_dim)

        pred_valid_points = valid_points(pred_poses)[0:1]
        quality_samples[post_fix].append(pred_valid_points)

quality_samples['gt'] = np.concatenate(quality_samples['gt'], axis=1)
for post_fix in args.post_fix:
    quality_samples[post_fix] = np.concatenate(quality_samples[post_fix], axis=1)

print('gt:', quality_samples['gt'].shape)
quality_samples['gt'] = quality_samples['gt'].tolist()
for post_fix in args.post_fix:
    print(post_fix, ':', quality_samples[post_fix].shape)
    quality_samples[post_fix] = quality_samples[post_fix].tolist()

save_dir = '../../experiments/'
os.makedirs(save_dir, exist_ok=True)
save_name = os.path.join(save_dir, 'quality_samples_%s.json'%(speaker))
with open(save_name, 'w') as f:
    json.dump(quality_samples, f)

