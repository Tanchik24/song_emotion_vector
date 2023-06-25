import numpy as np
import librosa
import torch

def cut_music(music:np.ndarray, sr:int):
    music_length = music.shape[0] // sr
    frame_duration = 1
    frame_length = int(frame_duration * sr)
    music = librosa.util.frame(music, frame_length=frame_length, hop_length=frame_length, axis=0)
    return music 


def normalize_mfccs(mfcc):
    mean = torch.tensor(13.6779, dtype=torch.float)
    std = torch.tensor(53.4601, dtype=torch.float)
    return (mfcc - mean) / std


def get_mfcc(music, sr):
  """
    Calculate the Mel-frequency cepstral coefficients (MFCC) of an audio signal.

    Args:
        music (ndarray): Audio signal.
        sr (int): Sampling rate of the audio signal.

    Returns:
        ndarray: MFCC of the audio signal.
    """
  sr = sr / 1000
  n_fft = int(30 * sr)
  hop_length = int(10 * sr)
  mfcc = librosa.feature.mfcc(y=music, sr=sr, n_mfcc=20, n_fft= n_fft, hop_length=hop_length)
  return mfcc


def make_mono_sound(song):
    if song.shape[1] > 1:
        song = librosa.to_mono(song.T)
    return song


def preprocess_song(song, sr):
    mfcc = get_mfcc(song, sr)
    mfcc = torch.tensor(mfcc, dtype=torch.float)
    mfcc = normalize_mfccs(mfcc)
    mfcc = torch.unsqueeze(mfcc, 0)
    return mfcc