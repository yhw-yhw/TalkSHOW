import json
import textgrid as tg
import numpy as np

def get_parameter_size(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

def denormalize(kps, data_mean, data_std):
    '''
    kps: (B, T, C)
    '''
    data_std = data_std.reshape(1, 1, -1)
    data_mean = data_mean.reshape(1, 1, -1)
    return (kps * data_std) + data_mean

def normalize(kps, data_mean, data_std):
    '''
    kps: (B, T, C)
    '''
    data_std = data_std.squeeze().reshape(1, 1, -1)
    data_mean = data_mean.squeeze().reshape(1, 1, -1)

    return (kps-data_mean) / data_std

def parse_audio(textgrid_file):
    '''a demo implementation'''
    words=['but', 'as', 'to', 'that', 'with', 'of', 'the', 'and', 'or', 'not', 'which', 'what', 'this', 'for', 'because', 'if', 'so', 'just', 'about', 'like', 'by', 'how', 'from', 'whats', 'now', 'very', 'that', 'also', 'actually', 'who', 'then', 'well', 'where', 'even', 'today', 'between', 'than', 'when']
    txt=tg.TextGrid.fromFile(textgrid_file)
    
    total_time=int(np.ceil(txt.maxTime))
    code_seq=np.zeros(total_time)
    
    word_level=txt[0]
    
    for i in range(len(word_level)):
        start_time=word_level[i].minTime
        end_time=word_level[i].maxTime
        mark=word_level[i].mark
        
        if mark in words:
            start=int(np.round(start_time))
            end=int(np.round(end_time))
            
            if start >= len(code_seq) or end >= len(code_seq):
                code_seq[-1] = 1
            else:
                code_seq[start]=1
    
    return code_seq


def get_path(model_name, model_type):
    if model_name == 's2g_body_pixel':
        if model_type == 'mfcc':
            return './experiments/2022-10-09-smplx_S2G-body-pixel-aud-3p/ckpt-99.pth'
        elif model_type == 'wv2':
            return './experiments/2022-10-28-smplx_S2G-body-pixel-wv2-sg2/ckpt-99.pth'
        elif model_type == 'random':
            return './experiments/2022-10-09-smplx_S2G-body-pixel-random-3p/ckpt-99.pth'
        elif model_type == 'wbhmodel':
            return './experiments/2022-11-02-smplx_S2G-body-pixel-w-bhmodel/ckpt-99.pth'
        elif model_type == 'wobhmodel':
            return './experiments/2022-11-02-smplx_S2G-body-pixel-wo-bhmodel/ckpt-99.pth'
    elif model_name == 's2g_body':
        if model_type == 'a+m-vae':
            return './experiments/2022-10-19-smplx_S2G-body-audio-motion-vae/ckpt-99.pth'
        elif model_type == 'a-vae':
            return './experiments/2022-10-18-smplx_S2G-body-audiovae/ckpt-99.pth'
        elif model_type == 'a-ed':
            return './experiments/2022-10-18-smplx_S2G-body-audioae/ckpt-99.pth'
    elif model_name == 's2g_LS3DCG':
        return './experiments/2022-10-19-smplx_S2G-LS3DCG/ckpt-99.pth'
    elif model_name == 's2g_body_vq':
        if model_type == 'n_com_1024':
            return './experiments/2022-10-29-smplx_S2G-body-vq-cn1024/ckpt-99.pth'
        elif model_type == 'n_com_2048':
            return './experiments/2022-10-29-smplx_S2G-body-vq-cn2048/ckpt-99.pth'
        elif model_type == 'n_com_4096':
            return './experiments/2022-10-29-smplx_S2G-body-vq-cn4096/ckpt-99.pth'
        elif model_type == 'n_com_8192':
            return './experiments/2022-11-02-smplx_S2G-body-vq-cn8192/ckpt-99.pth'
        elif model_type == 'n_com_16384':
            return './experiments/2022-11-02-smplx_S2G-body-vq-cn16384/ckpt-99.pth'
        elif model_type == 'n_com_170000':
            return './experiments/2022-10-30-smplx_S2G-body-vq-cn170000/ckpt-99.pth'
        elif model_type == 'com_1024':
            return './experiments/2022-10-29-smplx_S2G-body-vq-composition/ckpt-99.pth'
        elif model_type == 'com_2048':
            return './experiments/2022-10-31-smplx_S2G-body-vq-composition2048/ckpt-99.pth'
        elif model_type == 'com_4096':
            return './experiments/2022-10-31-smplx_S2G-body-vq-composition4096/ckpt-99.pth'
        elif model_type == 'com_8192':
            return './experiments/2022-11-02-smplx_S2G-body-vq-composition8192/ckpt-99.pth'
        elif model_type == 'com_16384':
            return './experiments/2022-11-02-smplx_S2G-body-vq-composition16384/ckpt-99.pth'


def get_dpath(model_name, model_type):
    if model_name == 's2g_body_pixel':
        if model_type == 'audio':
            return './experiments/2022-10-26-smplx_S2G-d-pixel-aud/ckpt-9.pth'
        elif model_type == 'wv2':
            return './experiments/2022-11-04-smplx_S2G-d-pixel-wv2/ckpt-9.pth'
        elif model_type == 'random':
            return './experiments/2022-10-26-smplx_S2G-d-pixel-random/ckpt-9.pth'
        elif model_type == 'wbhmodel':
            return './experiments/2022-11-10-smplx_S2G-hD-wbhmodel/ckpt-9.pth'
            # return './experiments/2022-11-05-smplx_S2G-d-pixel-wbhmodel/ckpt-9.pth'
        elif model_type == 'wobhmodel':
            return './experiments/2022-11-10-smplx_S2G-hD-wobhmodel/ckpt-9.pth'
            # return './experiments/2022-11-05-smplx_S2G-d-pixel-wobhmodel/ckpt-9.pth'
    elif model_name == 's2g_body':
        if model_type == 'a+m-vae':
            return './experiments/2022-10-26-smplx_S2G-d-audio+motion-vae/ckpt-9.pth'
        elif model_type == 'a-vae':
            return './experiments/2022-10-26-smplx_S2G-d-audio-vae/ckpt-9.pth'
        elif model_type == 'a-ed':
            return './experiments/2022-10-26-smplx_S2G-d-audio-ae/ckpt-9.pth'
    elif model_name == 's2g_LS3DCG':
        return './experiments/2022-10-26-smplx_S2G-d-ls3dcg/ckpt-9.pth'