import csv
import os
import pandas as pd
import librosa
import soundfile as sf

def save_csv(music, music_name, sr, arousal_sec, valence_sec, counter):
    music = music[counter]
    sf.write(f'music_project/data/interim/deam/music/{music_name}_{counter}.wav', music, sr)
    with open('music_project/data/interim/deam/annotation.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerows([[f'{music_name}_{counter}', arousal_sec, valence_sec]])


def make_deam_dataset(song_dir, arousal_dir, valence_dir):

    with open('music_project/data/interim/deam', 'w') as file:
        writer = csv.writer(file)
        writer.writerows([['song_id', 'arousal', 'valence']])

    arousal = pd.read_csv(arousal_dir)
    valence = pd.read_csv(valence_dir)
    for elem in sorted(os.listdir(song_dir), key=lambda x: int(x[:x.index('.')])):
        music, sr = librosa.load(f'{song_dir}/{elem}', mono=True, sr=44100)
        start_time = librosa.time_to_samples(15, sr=sr)
        music = music[start_time:]
        music_length = music.shape[0] // sr
        if music_length == 0:
            continue
        frame_duration = 1
        frame_length = int(frame_duration * sr)
        music = librosa.util.frame(music, frame_length=frame_length, hop_length=frame_length, axis=0)
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
            save_csv(music, music_name, sr, arousal_sec, valence_sec, counter)
