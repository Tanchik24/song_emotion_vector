import csv
import soundfile as sf
import os
import librosa
import pandas as pd


def save_csv(music, music_name, sr, arousal_sec, valence_sec, counter):
    music = music[counter]
    sf.write(f'music_project/data/interim/PMEmo/{music_name}_{counter}.wav', music, sr)
    with open('music_project/data/interim/PMEmo/annotation.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerows([[f'{music_name}_{counter}', arousal_sec, valence_sec]])


def make_pmemo_dataset(song_dir, annotaion_dir):
    with open('music_project/data/interim/PMEmo/annotation.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows([['song_id', 'arousal', 'valence']])
    df = pd.read_csv(annotaion_dir)
    for elem in sorted(os.listdir(song_dir), key=lambda x: int(x[:x.index('.')])):
        music, sr = librosa.load(f'{song_dir}/{elem}', mono=True, sr=None)
        start_time = librosa.time_to_samples(15, sr=sr)
        music = music[start_time:]
        music_length = music.shape[0] // sr
        if music_length == 0:
            continue
        frame_duration = 1
        frame_length = int(frame_duration * sr)
        music = librosa.util.frame(music, frame_length=frame_length, hop_length=frame_length, axis=0)
        music_name = int(elem.split('.')[0])

        arousal = list(df[df['musicId'] == music_name]['Arousal(mean)'].values)
        valence = list(df[df['musicId'] == music_name]['Valence(mean)'].values)

        for counter, index in enumerate(range(0, len(arousal), 2)):
            if (counter == len(music)) or (index + 1 == len(arousal)):
                continue
            arousal_sec = (arousal[index] + arousal[index + 1]) / 2
            valence_sec = (valence[index] + valence[index + 1]) / 2
            save_csv(music, music_name, sr, arousal_sec, valence_sec, counter)
