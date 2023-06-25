import librosa
import shutil
import os
import numpy as np
import pandas as pd


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
    mfcc = librosa.feature.mfcc(y=music, sr=sr, n_mfcc=20,
                                n_fft=n_fft, hop_length=hop_length)
    return mfcc


def create_folder(folder_path):
    """
      Create a new folder at the specified path.

      If a folder already exists at the given path, it will be deleted first and then recreated.

      Args:
          folder_path (str): The path of the folder to create.

      Returns:
          None
      """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)
    else:
        os.mkdir(folder_path)


def make_mfcc_dataset(music_dir, annot_dir, dir):
    """
      Creates an MFCC representation for each audio file and saves it as a NumPy array in the
      'data/processed/mfcc' directory.

      Args:
          music_dir (str): The directory containing the audio files.
          annot_dir (str): The path to the annotation CSV file.

      Returns:
          None
      """
    create_folder(f'{dir}/mfcc')
    annot_df = pd.read_csv(annot_dir)
    for song in list(annot_df['song_id'].values):
        music, sr = librosa.load(os.path.join(music_dir, song + '.wav'), sr=None, mono=True)
        mfcc = get_mfcc(music, sr)
        if mfcc is None:
            continue
        np.save(f'{dir}/mfcc/{song}.npy', mfcc)
