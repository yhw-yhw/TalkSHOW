import random

import cv2
import os

import tempfile
import threading
from subprocess import call

import numpy as np
from scipy.io import wavfile
import pyrender

import librosa

from tqdm import tqdm

# import open3d as o3d
from data_utils.utils import load_wav_old
from voca.rendering import render_mesh_helper


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def get_sen(i, num_video, i_frame, pos):
    if num_video == 1:
        sen = 'GT'
    elif num_video == 2:
        if i == 0:
            if pos == 1:
                sen = 'A'
            elif pos == 2:
                sen = 'B'
            else:
                sen = 'GT'
        else:
            if pos == 1:
                sen = 'B'
            elif pos == 2:
                sen = 'A'
            else:
                sen = 'result'
    elif num_video == 3:
        if i == 0:
            sen = 'sample1'
        elif i == 1:
            sen = 'interpolation'
        else:
            sen = 'sample2'
    elif num_video == 9 or num_video == 16:
        if i == 0:
            sen = 'frame '+str(i_frame)
        else:
            sen = 'sample' + str(i)
    elif num_video == 12:
        if i == 0:
            sen = 'sample1'
        elif i < 11:
            sen = 'interpolation' + str(i)
        else:
            sen = 'sample2'

    return sen


def add_image_text(img, text, color=(0,0,255), w=800, h=800):
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, 8, 2)[0]
    textX = (img.shape[1] - textsize[0]) // 2
    textY = textsize[1] + 10
    # img = img.copy()
    # a = img * 255
    # img = a.transpose(1, 2, 0).astype(np.uint8).copy()
    # cv2.putText(img, '%s' % (text), (textX, textY), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # w = int(text)

    # img = img.transpose(1, 2, 0)
    img = np.require(img, dtype='f4', requirements=['O', 'W'])
    img.flags.writeable = True
    img1 = img.copy()
    img1 = cv2.putText(img1, '%s' % (text), (100, 100), font, 4, color, 2, 1)
    img1 = cv2.rectangle(img1, (0, 0), (w, h), color, thickness=3, )

    # img1 = img1.transpose(2, 0, 1)

    return img1


class RenderTool():
    def __init__(self, out_path):
        path = os.path.join(os.getcwd(), 'visualise/smplx/SMPLX_NEUTRAL.npz')
        model_data = np.load(path, allow_pickle=True)
        data_struct = Struct(**model_data)
        self.f = data_struct.f
        self.out_path = out_path
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

    def _render_sequences(self, cur_wav_file, v_list, j=-1, stand=False, face=False, whole_body=False, run_in_parallel=False, transcript=None):
        # import sys
        # if sys.platform == 'win32':
        symbol = '/'
        # else:
        #     symbol = '\\'
        print("Render {} {} sequence.".format(cur_wav_file.split(symbol)[-2],cur_wav_file.split(symbol)[-1]))
        if run_in_parallel:
            thread = threading.Thread(target=self._render_helper, args=(cur_wav_file, v_list))
            thread.start()
            thread.join()
        else:
            # directory = os.path.join(self.out_path, cur_wav_file.split(symbol)[-2])
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            # video_fname = os.path.join(directory, '%s.mp4' % cur_wav_file.split(symbol)[-1].split('.')[-2])
            directory = os.path.join(self.out_path, cur_wav_file.split(symbol)[2].split(symbol)[0])
            if not os.path.exists(directory):
                os.makedirs(directory)
            if j == -1:
                video_fname = os.path.join(directory, '%s.mp4' % cur_wav_file.split(symbol)[-1].split('.')[-2].split(symbol)[-1])
            elif j == -2:
                video_fname = os.path.join(directory, cur_wav_file.split(symbol)[-3]+'--%s.mp4' % cur_wav_file.split(symbol)[-1].split('.')[-2].split(symbol)[-1])
            else:
                video_fname = os.path.join(directory, str(j)+'_%s.mp4' % cur_wav_file.split(symbol)[-1].split('.')[-2].split(symbol)[-1])
            self._render_sequences_helper(video_fname, cur_wav_file, v_list, stand, face, whole_body, transcript)

    def _render_sequences_helper(self, video_fname, cur_wav_file, v_list, stand, face, whole_body, transcript):
        num_frames = v_list[0].shape[0]

        # dataset is inverse
        for v in v_list:
            v = v.reshape(v.shape[0], -1, 3)
            v[:, :, 1] = -v[:, :, 1]
            v[:, :, 2] = -v[:, :, 2]
        viewport_height = 800
        z_offset = 1.0
        num_video = len(v_list)
        assert num_video in [1, 2, 3, 9, 12, 16, 18]
        if num_video == 1:
            width, height = 800, 800
        elif num_video == 2:
            width, height = 1600, 800
        elif num_video == 3:
            width, height = 2400, 800
        elif num_video == 9:
            width, height = 2400, 2400
        elif num_video == 12:
            width, height = 3200, 2400
        elif num_video == 16:
            width, height = 3200, 3200
        elif num_video == 18:
            width, height = 4800, 2400

        if whole_body:
            width, height = 800, 1440
            viewport_height = 1440
            z_offset = 1.8

        sr = 22000
        audio, sr = librosa.load(cur_wav_file, sr=16000)
        tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.wav', dir=os.path.dirname(video_fname))
        tmp_audio_file.close()
        wavfile.write(tmp_audio_file.name, sr, audio)
        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=os.path.dirname(video_fname))
        tmp_video_file.close()
        if int(cv2.__version__[0]) < 3:
            print('cv2 < 3')
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), 30, (width, height), True)
        else:
            print('cv2 >= 3')
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), True)

        center = np.mean(v_list[0][0], axis=0)

        r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=viewport_height)

        # random exchange the position of our method and SG3D
        # pos = random.randint(1, 2)
        # video_fname = list(video_fname)
        # video_fname.insert(-4, str(pos))
        # video_fname = ''.join(video_fname)
        pos = 1

        for i_frame in tqdm(range(num_frames)):
            # pyrender.Viewer(scene)
            cur_img = []
            for i in range(len(v_list)):
                if face:
                    img = render_mesh_helper((v_list[i][i_frame], self.f), center,
                                             r=r, xmag=0.15, y=1, z=1.0, camera='o')
                else:
                    img = render_mesh_helper((v_list[i][i_frame], self.f), center, camera='o', r=r, y=0.7, z_offset=z_offset)
                # sen = get_sen(i, num_video, i_frame, pos)
                # if transcript is not None:
                #     sen = str(int(transcript[i_frame].item()))
                # else:
                #     sen = ' '
                # img = add_image_text(img, sen)
                cur_img.append(img)

            if num_video == 1:
                final_img = cur_img[0].astype(np.uint8)
            elif num_video == 2:
                final_img = np.hstack((cur_img[0], cur_img[1])).astype(np.uint8)
            elif num_video == 3:
                final_img = np.hstack((cur_img[0], cur_img[1], cur_img[2])).astype(np.uint8)
            elif num_video == 9:
                img_vert_0 = np.hstack((cur_img[0], cur_img[1], cur_img[2])).astype(np.uint8)
                img_vert_1 = np.hstack((cur_img[3], cur_img[4], cur_img[5])).astype(np.uint8)
                img_vert_2 = np.hstack((cur_img[6], cur_img[7], cur_img[8])).astype(np.uint8)
                final_img = np.vstack((img_vert_0, img_vert_1, img_vert_2)).astype(np.uint8)
            elif num_video == 12:
                img_vert_0 = np.hstack((cur_img[0], cur_img[1], cur_img[2], cur_img[3])).astype(np.uint8)
                img_vert_1 = np.hstack((cur_img[4], cur_img[5], cur_img[6], cur_img[7])).astype(np.uint8)
                img_vert_2 = np.hstack((cur_img[8], cur_img[9], cur_img[10], cur_img[11])).astype(np.uint8)
                final_img = np.vstack((img_vert_0, img_vert_1, img_vert_2)).astype(np.uint8)
            elif num_video == 16:
                img_vert_0 = np.hstack((cur_img[0], cur_img[1], cur_img[2], cur_img[3])).astype(np.uint8)
                img_vert_1 = np.hstack((cur_img[4], cur_img[5], cur_img[6], cur_img[7])).astype(np.uint8)
                img_vert_2 = np.hstack((cur_img[8], cur_img[9], cur_img[10], cur_img[11])).astype(np.uint8)
                img_vert_3 = np.hstack((cur_img[12], cur_img[13], cur_img[14], cur_img[15])).astype(np.uint8)
                final_img = np.vstack((img_vert_0, img_vert_1, img_vert_2, img_vert_3)).astype(np.uint8)
            elif num_video == 18:
                img_vert_0 = np.hstack((cur_img[0], cur_img[1], cur_img[2], cur_img[3], cur_img[4], cur_img[5])).astype(np.uint8)
                img_vert_1 = np.hstack((cur_img[6], cur_img[7], cur_img[8], cur_img[9], cur_img[10], cur_img[11])).astype(np.uint8)
                img_vert_2 = np.hstack((cur_img[12], cur_img[13], cur_img[14], cur_img[15], cur_img[16], cur_img[17])).astype(
                    np.uint8)
                final_img = np.vstack((img_vert_0, img_vert_1, img_vert_2)).astype(np.uint8)
            # final_img = add_image_text(final_img, 'frame'+str(i_frame), w=width, h=height)
            writer.write(final_img)
        writer.release()

        cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {2}'.format(
            tmp_audio_file.name, tmp_video_file.name, video_fname)).split()
        # cmd = ('ffmpeg' + '-i {0} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {1}'.format(
        #     tmp_video_file.name, video_fname)).split()
        call(cmd)
        os.remove(tmp_audio_file.name)
        os.remove(tmp_video_file.name)

    def _render_continuity(self, cur_wav_file, pred, frame, run_in_parallel=False):
        print("Render {} {} sequence.".format(cur_wav_file.split(symbol)[-2],cur_wav_file.split(symbol)[-1]))
        if run_in_parallel:
            thread = threading.Thread(target=self._render_helper, args=(cur_wav_file, pred))
            thread.start()
            thread.join()
        else:
            self._render_helper2(cur_wav_file, pred, frame)

    def _render_helper2(self, cur_wav_file, pred, frame):
        directory = os.path.join(self.out_path, cur_wav_file.split('/')[2].split(symbol)[0])
        if not os.path.exists(directory):
            os.makedirs(directory)
        video_fname = os.path.join(directory, '%s.mp4' % cur_wav_file.split(symbol)[-1].split('.')[-2].split('/')[-1])
        self._render_sequences_helper2(video_fname, cur_wav_file, pred, frame)

    def _render_sequences_helper2(self, video_fname, cur_wav_file, pred, frame):

        num_frames = pred.shape[0]
        pred = pred.reshape(pred.shape[0], -1, 3)

        pred[:, :, 1] = -pred[:, :, 1]
        pred[:, :, 2] = -pred[:, :, 2]

        sr = 22000
        audio, sr = load_wav_old(cur_wav_file, sr=sr)
        tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.wav', dir=os.path.dirname(video_fname))
        tmp_audio_file.close()
        wavfile.write(tmp_audio_file.name, sr, audio)
        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=os.path.dirname(video_fname))
        tmp_video_file.close()
        if int(cv2.__version__[0]) < 3:
            print('cv2 < 3')
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), 15, (190, 800), True)
        else:
            print('cv2 >= 3')
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (800, 800), True)

        center = np.mean(pred[0], axis=0)

        r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)

        for i_frame in tqdm(range(num_frames)):
            if i_frame < frame:
                sen = 'sequence 1'
                color = (0,255,0)
            else:
                sen = 'sequence 2'
                color = (0, 0, 255)
            pred_img = render_mesh_helper(Mesh(pred[i_frame], self.template_mesh.f), center, camera='o',r=r, y=0.7)
            pred_img = add_image_text(pred_img, sen, color)
            pred_img = pred_img.astype(np.uint8)
            writer.write(pred_img)
        writer.release()

        cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {2}'.format(
            tmp_audio_file.name, tmp_video_file.name, video_fname)).split()
        call(cmd)
        os.remove(tmp_audio_file.name)
        os.remove(tmp_video_file.name)