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

precision_list=[]
recall_list=[]
accuracy_list=[]

for aud in tqdm(test_audios):
    base_name = os.path.splitext(aud)[0]
    gt_path = get_full_path(aud, speaker, 'val')
    _, gt_poses, _ = get_gts(gt_path)
    if gt_poses.shape[0] < 50:
        continue
    gt_poses = gt_poses[np.newaxis,...]
    # print(gt_poses.shape)#(seq_len, 135*2)pose, lhand, rhand, face
    for post_fix in args.post_fix:
        pred_path = base_name + '_'+post_fix+'.json'
        pred_poses = np.array(json.load(open(pred_path)))
        # print(pred_poses.shape)#(B, seq_len, 108)
        pred_poses = cvt25(pred_poses, gt_poses)
        # print(pred_poses.shape)#(B, seq, pose_dim)

        gt_valid_points = valid_points(gt_poses)
        pred_valid_points = valid_points(pred_poses)

        # print(gt_valid_points.shape, pred_valid_points.shape)

        gt_mode_transition_seq = mode_transition_seq(gt_valid_points, speaker)#(B, N)
        pred_mode_transition_seq = mode_transition_seq(pred_valid_points, speaker)#(B, N)

        # baseline = np.random.randint(0, 2, size=pred_mode_transition_seq.shape)
        # pred_mode_transition_seq = baseline
        precision, recall, accuracy = mode_transition_consistency(pred_mode_transition_seq, gt_mode_transition_seq)
        precision_list.append(precision)
        recall_list.append(recall)
        accuracy_list.append(accuracy)
print(len(precision_list), len(recall_list), len(accuracy_list))
precision_list = np.mean(precision_list)
recall_list = np.mean(recall_list)
accuracy_list = np.mean(accuracy_list)

print('precision, recall, accu:', precision_list, recall_list, accuracy_list)
