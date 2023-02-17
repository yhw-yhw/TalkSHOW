import os
from glob import glob
import numpy as np
import json
from matplotlib import pyplot as plt
import pandas as pd
def get_gts(clip):    
    '''
    clip: abs path to the clip dir
    '''
    keypoints_files = sorted(glob(os.path.join(clip, 'keypoints_new/person_1')+'/*.json'))
    
    upper_body_points = list(np.arange(0, 25))
    poses = [] 
    confs = []
    neck_to_nose_len = []
    mean_position = []
    for kp_file in keypoints_files:
        kp_load = json.load(open(kp_file, 'r'))['people'][0]
        posepts = kp_load['pose_keypoints_2d']
        lhandpts = kp_load['hand_left_keypoints_2d']
        rhandpts = kp_load['hand_right_keypoints_2d']
        facepts = kp_load['face_keypoints_2d']

        neck = np.array(posepts).reshape(-1,3)[1]
        nose = np.array(posepts).reshape(-1,3)[0]
        x_offset = abs(neck[0]-nose[0])
        y_offset = abs(neck[1]-nose[1])
        neck_to_nose_len.append(y_offset)
        mean_position.append([neck[0],neck[1]])

        keypoints=np.array(posepts+lhandpts+rhandpts+facepts).reshape(-1,3)[:,:2]

        upper_body = keypoints[upper_body_points, :]
        hand_points = keypoints[25:, :]
        keypoints = np.vstack([upper_body, hand_points])

        poses.append(keypoints)

    if len(neck_to_nose_len) > 0:
        scale_factor = np.mean(neck_to_nose_len)
    else:
        raise ValueError(clip)
    mean_position = np.mean(np.array(mean_position), axis=0)
    
    unlocalized_poses = np.array(poses).copy()
    localized_poses = []
    for i in range(len(poses)):
        keypoints = poses[i]
        neck = keypoints[1].copy()

        keypoints[:, 0] = (keypoints[:, 0] - neck[0]) / scale_factor
        keypoints[:, 1] = (keypoints[:, 1] - neck[1]) / scale_factor
        localized_poses.append(keypoints.reshape(-1))
        
    localized_poses=np.array(localized_poses)
    return unlocalized_poses, localized_poses, (scale_factor, mean_position)

def get_full_path(wav_name, speaker, split):
    '''
    get clip path from aud file
    '''
    wav_name = os.path.basename(wav_name)
    wav_name = os.path.splitext(wav_name)[0]
    clip_name, vid_name = wav_name[:10], wav_name[11:]

    full_path = os.path.join('pose_dataset/videos/', speaker, 'clips', vid_name, 'images/half', split, clip_name)

    assert os.path.isdir(full_path), full_path

    return full_path

def smooth(res):
    '''
    res: (B, seq_len, pose_dim)
    '''
    window = [res[:, 7, :], res[:, 8, :], res[:, 9, :], res[:, 10, :], res[:, 11, :], res[:, 12, :]]
    w_size=7
    for i in range(10, res.shape[1]-3):
        window.append(res[:, i+3, :])
        if len(window) > w_size:
            window = window[1:]
        
        if (i%25) in [22, 23, 24, 0, 1, 2, 3]:
            res[:, i, :] = np.mean(window, axis=1)
    
    return res

def cvt25(pred_poses, gt_poses=None):
    '''
    gt_poses: (1, seq_len, 270), 135 *2
    pred_poses: (B, seq_len, 108), 54 * 2
    '''
    if gt_poses is None:
        gt_poses = np.zeros_like(pred_poses)
    else:
        gt_poses = gt_poses.repeat(pred_poses.shape[0], axis=0)

    length = min(pred_poses.shape[1], gt_poses.shape[1])
    pred_poses = pred_poses[:, :length, :]
    gt_poses = gt_poses[:, :length, :]
    gt_poses = gt_poses.reshape(gt_poses.shape[0], gt_poses.shape[1], -1, 2)
    pred_poses = pred_poses.reshape(pred_poses.shape[0], pred_poses.shape[1], -1, 2)

    gt_poses[:, :, [1, 2, 3, 4, 5, 6, 7], :] = pred_poses[:, :, 1:8, :]
    gt_poses[:, :, 25:25+21+21, :] = pred_poses[:, :, 12:, :]
    
    return gt_poses.reshape(gt_poses.shape[0], gt_poses.shape[1], -1)

def hand_points(seq):
    '''
    seq: (B, seq_len, 135*2)
    hands only
    '''
    hand_idx = [1, 2, 3, 4,5 ,6,7] + list(range(25, 25+21+21))
    seq = seq.reshape(seq.shape[0], seq.shape[1], -1, 2)
    return seq[:, :, hand_idx, :].reshape(seq.shape[0], seq.shape[1], -1)

def valid_points(seq):
    '''
    hands with some head points
    '''
    valid_idx = [0, 1, 2, 3, 4,5 ,6,7, 8, 9, 10, 11] + list(range(25, 25+21+21))
    seq = seq.reshape(seq.shape[0], seq.shape[1], -1, 2)

    seq = seq[:, :, valid_idx, :].reshape(seq.shape[0], seq.shape[1], -1)
    assert seq.shape[-1] == 108, seq.shape
    return seq

def draw_cdf(seq, save_name='cdf.jpg', color='slatebule'):
    plt.figure()
    plt.hist(seq, bins=100, range=(0, 100), color=color)
    plt.savefig(save_name)

def to_excel(seq, save_name='res.xlsx'):
    '''
    seq: (T)
    '''
    df = pd.DataFrame(seq)
    writer = pd.ExcelWriter(save_name)
    df.to_excel(writer, 'sheet1')
    writer.save()
    writer.close()


if __name__ == '__main__':
    random_data = np.random.randint(0, 10, 100)
    draw_cdf(random_data)