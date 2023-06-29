import csv
import os
import pandas as pd
import librosa

from src.data.data_utils import create_csv, cut_music, save_csv


def make_deam_dataset(song_dir, arousal_dir, valence_dir):
    dir_ = '../data/interim/deam'
    create_csv(dir_)
    arousal = pd.read_csv(arousal_dir)
    valence = pd.read_csv(valence_dir)
    for elem in sorted(os.listdir(song_dir), key=lambda x: int(x[:x.index('.')])):
        results = cut_music(f'{song_dir}/{elem}')
        if results is None:
            continue
        music, sr = results
        music_name = int(elem.split('.')[0])

        arousal_df = arousal[arousal['song_id'] == music_name].drop(columns=['song_id', 'sample_15000ms'])
        arousal_df = list(arousal_df.dropna(axis=1).values[0])
        valence_df = valence[valence['song_id'] == music_name].drop(columns=['song_id', 'sample_15000ms'])
        valence_df = list(valence_df.dropna(axis=1).values[0])

        for counter, index in enumerate(range(0, len(arousal_df), 2)):
            if (counter == len(music)) or (index == len(arousal_df)) or (index + 1 == len(arousal_df)):
                continue
            if (counter == len(music)) or (index == len(valence_df)) or (index + 1 == len(valence_df)):
                continue
            arousal_sec = (arousal_df[index] + arousal_df[index + 1]) / 2
            valence_sec = (valence_df[index] + valence_df[index + 1]) / 2
            save_csv(music, music_name, dir_, sr, arousal_sec, valence_sec, counter)
