import numpy as np
# import librosa #has to do this cause librosa is not supported on my server
import python_speech_features
from scipy.io import wavfile
from scipy import signal
import librosa
import torch
import torchaudio as ta
import torchaudio.functional as ta_F
import torchaudio.transforms as ta_T
# import pyloudnorm as pyln


def load_wav_old(audio_fn, sr = 16000):
    sample_rate, sig = wavfile.read(audio_fn)
    if sample_rate != sr:
        result = int((sig.shape[0]) / sample_rate * sr)
        x_resampled = signal.resample(sig, result)
        x_resampled = x_resampled.astype(np.float64)
        return x_resampled, sr
    
    sig = sig / (2**15)
    return sig, sample_rate


def get_mfcc(audio_fn, eps=1e-6, fps=25, smlpx=False, sr=16000, n_mfcc=64, win_size=None):

    y, sr = librosa.load(audio_fn, sr=sr, mono=True)

    if win_size is None:
        hop_len=int(sr / fps)
    else:
        hop_len=int(sr / win_size)
        
    n_fft=2048

    C = librosa.feature.mfcc(
        y = y,
        sr = sr,
        n_mfcc = n_mfcc,
        hop_length = hop_len,
        n_fft = n_fft
    )

    if C.shape[0] == n_mfcc:
        C = C.transpose(1, 0)
    
    return C

    
def get_melspec(audio_fn, eps=1e-6, fps = 25, sr=16000, n_mels=64):
    raise NotImplementedError
    '''
    # y, sr = load_wav(audio_fn=audio_fn, sr=sr)
    
    # hop_len = int(sr / fps) 
    # n_fft = 2048

    # C = librosa.feature.melspectrogram(
    #     y = y, 
    #     sr = sr, 
    #     n_fft=n_fft, 
    #     hop_length=hop_len, 
    #     n_mels = n_mels, 
    #     fmin=0, 
    #     fmax=8000)
    

    # mask = (C == 0).astype(np.float)
    # C = mask * eps + (1-mask) * C

    # C = np.log(C)
    # #wierd error may occur here
    # assert not (np.isnan(C).any()), audio_fn
    # if C.shape[0] == n_mels:
    #     C = C.transpose(1, 0)

    # return C 
    '''

def extract_mfcc(audio,sample_rate=16000):
    mfcc = zip(*python_speech_features.mfcc(audio,sample_rate, numcep=64, nfilt=64, nfft=2048, winstep=0.04))
    mfcc = np.stack([np.array(i) for i in mfcc])
    return mfcc

def get_mfcc_psf(audio_fn, eps=1e-6, fps=25, smlpx=False, sr=16000, n_mfcc=64, win_size=None):
    y, sr = load_wav_old(audio_fn, sr=sr)

    if y.shape.__len__() > 1:
        y = (y[:,0]+y[:,1])/2

    if win_size is None:
        hop_len=int(sr / fps)
    else:
        hop_len=int(sr/ win_size)
        
    n_fft=2048 

    #hard coded for 25 fps
    if not smlpx:
        C = python_speech_features.mfcc(y, sr, numcep=n_mfcc, nfilt=n_mfcc, nfft=n_fft, winstep=0.04)
    else:
        C = python_speech_features.mfcc(y, sr, numcep=n_mfcc, nfilt=n_mfcc, nfft=n_fft, winstep=1.01/15)
    # if C.shape[0] == n_mfcc:
    #     C = C.transpose(1, 0)
    
    return C


def get_mfcc_psf_min(audio_fn, eps=1e-6, fps=25, smlpx=False, sr=16000, n_mfcc=64, win_size=None):
    y, sr = load_wav_old(audio_fn, sr=sr)

    if y.shape.__len__() > 1:
        y = (y[:, 0] + y[:, 1]) / 2
    n_fft = 2048

    slice_len = 22000 * 5
    slice = y.size // slice_len

    C = []

    for i in range(slice):
        if i != (slice - 1):
            feat = python_speech_features.mfcc(y[i*slice_len:(i+1)*slice_len], sr, numcep=n_mfcc, nfilt=n_mfcc, nfft=n_fft, winstep=1.01 / 15)
        else:
            feat = python_speech_features.mfcc(y[i * slice_len:], sr, numcep=n_mfcc, nfilt=n_mfcc, nfft=n_fft, winstep=1.01 / 15)

        C.append(feat)

    return C


def audio_chunking(audio: torch.Tensor, frame_rate: int = 30, chunk_size: int = 16000):
    """
    :param audio: 1 x T tensor containing a 16kHz audio signal
    :param frame_rate: frame rate for video (we need one audio chunk per video frame)
    :param chunk_size: number of audio samples per chunk
    :return: num_chunks x chunk_size tensor containing sliced audio
    """
    samples_per_frame = chunk_size // frame_rate
    padding = (chunk_size - samples_per_frame) // 2
    audio = torch.nn.functional.pad(audio.unsqueeze(0), pad=[padding, padding]).squeeze(0)
    anchor_points = list(range(chunk_size//2, audio.shape[-1]-chunk_size//2, samples_per_frame))
    audio = torch.cat([audio[:, i-chunk_size//2:i+chunk_size//2] for i in anchor_points], dim=0)
    return audio


def  get_mfcc_ta(audio_fn, eps=1e-6, fps=15, smlpx=False, sr=16000, n_mfcc=64, win_size=None, type='mfcc', am=None, am_sr=None, encoder_choice='mfcc'):
    if am is None:
        audio, sr_0 = ta.load(audio_fn)
        if sr != sr_0:
            audio = ta.transforms.Resample(sr_0, sr)(audio)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        n_fft = 2048
        if fps == 15:
            hop_length = 1467
        elif fps == 30:
            hop_length = 734
        win_length = hop_length * 2
        n_mels = 256
        n_mfcc = 64

        if type == 'mfcc':
            mfcc_transform = ta_T.MFCC(
                sample_rate=sr,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "n_mels": n_mels,
                    # "win_length": win_length,
                    "hop_length": hop_length,
                    "mel_scale": "htk",
                },
            )
            audio_ft = mfcc_transform(audio).squeeze(dim=0).transpose(0,1).numpy()
        elif type == 'mel':
            # audio = 0.01 * audio / torch.mean(torch.abs(audio))
            mel_transform = ta_T.MelSpectrogram(
                sample_rate=sr, n_fft=n_fft, win_length=None, hop_length=hop_length, n_mels=n_mels
            )
            audio_ft = mel_transform(audio).squeeze(0).transpose(0,1).numpy()
            # audio_ft = torch.log(audio_ft.clamp(min=1e-10, max=None)).transpose(0,1).numpy()
        elif type == 'mel_mul':
            audio = 0.01 * audio / torch.mean(torch.abs(audio))
            audio = audio_chunking(audio, frame_rate=fps, chunk_size=sr)
            mel_transform = ta_T.MelSpectrogram(
                sample_rate=sr, n_fft=n_fft, win_length=int(sr/20), hop_length=int(sr/100), n_mels=n_mels
            )
            audio_ft = mel_transform(audio).squeeze(1)
            audio_ft = torch.log(audio_ft.clamp(min=1e-10, max=None)).numpy()
    else:
        speech_array, sampling_rate = librosa.load(audio_fn, sr=16000)

        if encoder_choice == 'faceformer':
            # audio_ft = np.squeeze(am(speech_array, sampling_rate=16000).input_values).reshape(-1, 1)
            audio_ft = speech_array.reshape(-1, 1)
        elif encoder_choice == 'meshtalk':
            audio_ft = 0.01 * speech_array / np.mean(np.abs(speech_array))
        elif encoder_choice == 'onset':
            audio_ft = librosa.onset.onset_detect(y=speech_array, sr=16000, units='time').reshape(-1, 1)
        else:
            audio, sr_0 = ta.load(audio_fn)
            if sr != sr_0:
                audio = ta.transforms.Resample(sr_0, sr)(audio)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            n_fft = 2048
            if fps == 15:
                hop_length = 1467
            elif fps == 30:
                hop_length = 734
            win_length = hop_length * 2
            n_mels = 256
            n_mfcc = 64

            mfcc_transform = ta_T.MFCC(
                sample_rate=sr,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "n_mels": n_mels,
                    # "win_length": win_length,
                    "hop_length": hop_length,
                    "mel_scale": "htk",
                },
            )
            audio_ft = mfcc_transform(audio).squeeze(dim=0).transpose(0, 1).numpy()
    return audio_ft


def  get_mfcc_sepa(audio_fn, fps=15, sr=16000):
    audio, sr_0 = ta.load(audio_fn)
    if sr != sr_0:
        audio = ta.transforms.Resample(sr_0, sr)(audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    n_fft = 2048
    if fps == 15:
        hop_length = 1467
    elif fps == 30:
        hop_length = 734
    n_mels = 256
    n_mfcc = 64

    mfcc_transform = ta_T.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "n_mels": n_mels,
            # "win_length": win_length,
            "hop_length": hop_length,
            "mel_scale": "htk",
        },
    )
    audio_ft_0 = mfcc_transform(audio[0, :sr*2]).squeeze(dim=0).transpose(0,1).numpy()
    audio_ft_1 = mfcc_transform(audio[0, sr*2:]).squeeze(dim=0).transpose(0,1).numpy()
    audio_ft = np.concatenate((audio_ft_0, audio_ft_1), axis=0)
    return audio_ft, audio_ft_0.shape[0]


def get_mfcc_old(wav_file):
    sig, sample_rate = load_wav_old(wav_file)
    mfcc = extract_mfcc(sig)
    return mfcc


def smooth_geom(geom, mask: torch.Tensor = None, filter_size: int = 9, sigma: float = 2.0):
    """
    :param geom: T x V x 3 tensor containing a temporal sequence of length T with V vertices in each frame
    :param mask: V-dimensional Tensor containing a mask with vertices to be smoothed
    :param filter_size: size of the Gaussian filter
    :param sigma: standard deviation of the Gaussian filter
    :return: T x V x 3 tensor containing smoothed geometry (i.e., smoothed in the area indicated by the mask)
    """
    assert filter_size % 2 == 1, f"filter size must be odd but is {filter_size}"
    # Gaussian smoothing (low-pass filtering)
    fltr = np.arange(-(filter_size // 2), filter_size // 2 + 1)
    fltr = np.exp(-0.5 * fltr ** 2 / sigma ** 2)
    fltr = torch.Tensor(fltr) / np.sum(fltr)
    # apply fltr
    fltr = fltr.view(1, 1, -1).to(device=geom.device)
    T, V = geom.shape[1], geom.shape[2]
    g = torch.nn.functional.pad(
        geom.permute(2, 0, 1).view(V, 1, T),
        pad=[filter_size // 2, filter_size // 2], mode='replicate'
    )
    g = torch.nn.functional.conv1d(g, fltr).view(V, 1, T)
    smoothed = g.permute(1, 2, 0).contiguous()
    # blend smoothed signal with original signal
    if mask is None:
        return smoothed
    else:
        return smoothed * mask[None, :, None] + geom * (-mask[None, :, None] + 1)

if __name__ == '__main__':
    audio_fn = '../sample_audio/clip000028_tCAkv4ggPgI.wav'
    
    C = get_mfcc_psf(audio_fn)
    print(C.shape)

    C_2 = get_mfcc_librosa(audio_fn)
    print(C.shape)

    print(C)
    print(C_2)
    print((C == C_2).all())
    # print(y.shape, sr)
    # mel_spec = get_melspec(audio_fn)
    # print(mel_spec.shape)
    # mfcc = get_mfcc(audio_fn, sr = 16000)
    # print(mfcc.shape)
    # print(mel_spec.max(), mel_spec.min())
    # print(mfcc.max(), mfcc.min())