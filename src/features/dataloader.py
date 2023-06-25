import os
import torch
import pandas as pd
import numpy as np

class MusicDataset(Dataset):
  def __init__(self, annot_dir, mfcc_dir):
    self.mfcc_dir = mfcc_dir
    self.annot_df = pd.read_csv(annot_dir)
    self.music_name = list(self.annot_df['song_id'].values)

  def __len__(self):
    return len(self.music_name)

  def normalize_mfccs(self, mfcc):
    mean = torch.tensor(10.8789, dtype=torch.float)
    std = torch.tensor(61.9204, dtype=torch.float)
    return (mfcc - mean) / std

  def __getitem__(self, idx):
    series = self.annot_df.iloc[idx]
    mfcc = self.normalize_mfccs(torch.tensor(np.load(os.path.join(self.mfcc_dir, f"{series['song_id']}.npy")), dtype=torch.float))
    arousal = torch.tensor(series['arousal'], dtype=torch.float)
    valence = torch.tensor(series['valence'], dtype=torch.float)
    return mfcc, arousal, valence