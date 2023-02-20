import os
import pickle
from tqdm import tqdm
import shutil
import torch
import numpy as np
import librosa
import random

speakers = ['seth', 'conan', 'oliver', 'chemistry']
data_root = "../ExpressiveWholeBodyDatasetv1.0/"
split = 'train'



def split_list(full_list,shuffle=False,ratio=0.2):
    n_total = len(full_list)
    offset_0 = int(n_total * ratio)
    offset_1 = int(n_total * ratio * 2)
    if n_total==0 or offset_1<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_0 = full_list[:offset_0]
    sublist_1 = full_list[offset_0:offset_1]
    sublist_2 = full_list[offset_1:]
    return sublist_0, sublist_1, sublist_2


def moveto(list, file):
    for f in list:
        before, after = '/'.join(f.split('/')[:-1]), f.split('/')[-1]
        new_path = os.path.join(before, file)
        new_path = os.path.join(new_path, after)
        # os.makedirs(new_path)
        # os.path.isdir(new_path)
        # shutil.move(f, new_path)

        #转移到新目录
        shutil.copytree(f, new_path)
        #删除原train里的文件
        shutil.rmtree(f)
    return None


def read_pkl(data):
    betas = np.array(data['betas'])

    jaw_pose = np.array(data['jaw_pose'])
    leye_pose = np.array(data['leye_pose'])
    reye_pose = np.array(data['reye_pose'])
    global_orient = np.array(data['global_orient']).squeeze()
    body_pose = np.array(data['body_pose_axis'])
    left_hand_pose = np.array(data['left_hand_pose'])
    right_hand_pose = np.array(data['right_hand_pose'])

    full_body = np.concatenate(
        (jaw_pose, leye_pose, reye_pose, global_orient, body_pose, left_hand_pose, right_hand_pose), axis=1)

    expression = np.array(data['expression'])
    full_body = np.concatenate((full_body, expression), axis=1)

    if (full_body.shape[0] < 90) or (torch.isnan(torch.from_numpy(full_body)).sum() > 0):
        return 1
    else:
        return 0


for speaker_name in speakers:
    speaker_root = os.path.join(data_root, speaker_name)

    videos = [v for v in os.listdir(speaker_root)]
    print(videos)

    haode = huaide = 0
    total_seqs = []

    for vid in tqdm(videos, desc="Processing training data of {}......".format(speaker_name)):
    # for vid in videos:
        source_vid = vid
        vid_pth = os.path.join(speaker_root, source_vid)
        # vid_pth = os.path.join(speaker_root, source_vid, 'images/half', split)
        t = os.path.join(speaker_root, source_vid, 'test')
        v = os.path.join(speaker_root, source_vid, 'val')

        # if os.path.exists(t):
        #     shutil.rmtree(t)
        # if os.path.exists(v):
        #     shutil.rmtree(v)
        try:
            seqs = [s for s in os.listdir(vid_pth)]
        except:
            continue
        # if len(seqs) == 0:
        #     shutil.rmtree(os.path.join(speaker_root, source_vid))
            # None
        for s in seqs:
            quality = 0
            total_seqs.append(os.path.join(vid_pth,s))
            seq_root = os.path.join(vid_pth, s)
            key = seq_root  # correspond to clip******
            audio_fname = os.path.join(speaker_root, source_vid, s, '%s.wav' % (s))

            # delete the data without audio or the audio file could not be read
            if os.path.isfile(audio_fname):
                try:
                    audio = librosa.load(audio_fname)
                except:
                    # print(key)
                    shutil.rmtree(key)
                    huaide = huaide + 1
                    continue
            else:
                huaide = huaide + 1
                # print(key)
                shutil.rmtree(key)
                continue

            # check motion file
            motion_fname = os.path.join(speaker_root, source_vid, s, '%s.pkl' % (s))
            try:
                f = open(motion_fname, 'rb+')
            except:
                shutil.rmtree(key)
                huaide = huaide + 1
                continue

            data = pickle.load(f)
            w = read_pkl(data)
            f.close()
            quality = quality + w

            if w == 1:
                shutil.rmtree(key)
                # print(key)
                huaide = huaide + 1
                continue

            haode = haode + 1

    print("huaide:{}, haode:{}, total_seqs:{}".format(huaide, haode, total_seqs.__len__()))

for speaker_name in speakers:
    speaker_root = os.path.join(data_root, speaker_name)

    videos = [v for v in os.listdir(speaker_root)]
    print(videos)

    haode = huaide = 0
    total_seqs = []

    for vid in tqdm(videos, desc="Processing training data of {}......".format(speaker_name)):
        # for vid in videos:
        source_vid = vid
        vid_pth = os.path.join(speaker_root, source_vid)
        try:
            seqs = [s for s in os.listdir(vid_pth)]
        except:
            continue
        for s in seqs:
            quality = 0
            total_seqs.append(os.path.join(vid_pth, s))
    print("total_seqs:{}".format(total_seqs.__len__()))
    # split the dataset
    test_list, val_list, train_list = split_list(total_seqs, True, 0.1)
    print(len(test_list), len(val_list), len(train_list))
    moveto(train_list, 'train')
    moveto(test_list, 'test')
    moveto(val_list, 'val')

