import soundfile as sf
import librosa
import gdown
import pandas as pd
import csv
import os
import shutil
from src.features.features_utils import make_mono_sound


def save_csv(music, music_name, dir_, sr, arousal_sec, valence_sec, counter):
    """
        Saves a music file and its corresponding annotation to a CSV file.

        Args:
            music (numpy.ndarray): The audio data of the music.
            music_name (int): The name of the music.
            dir_ (str): The directory path where the files will be saved (pememo or deam).
            sr (int): The sample rate of the music.
            arousal_sec (float): The arousal value in seconds.
            valence_sec (float): The valence value in seconds.
            counter (int): The counter value used for naming the files.

        Returns:
            None
        """
    music = music[counter]
    sf.write(f'{dir_}/music/{music_name}_{counter}.wav', music, sr)
    with open(f'{dir_}/annotation.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerows([[f'{music_name}_{counter}', arousal_sec, valence_sec]])


def create_csv(dir_):
    """
        Creates a new CSV file for storing song annotations.

        Args:
            dir_ (str): The directory path where the CSV file will be created.

        Returns:
            None
        """
    with open(f'{dir_}/annotation.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows([['song_id', 'arousal', 'valence']])


def cut_music(song_dir):
    """
        Cuts a music file starting from 15 seconds and divides it into 1 sec frames.

        Args:
            song_dir (str): The directory path of the music file.

        Returns:
            tuple: A tuple containing the cut music and its sample rate (music, sr).
        """
    music, sr = sf.read(song_dir)
    music = make_mono_sound(music)
    start_time = librosa.time_to_samples(15, sr=sr)
    music = music[start_time:]
    music_length = music.shape[0] // sr
    if music_length == 0:
        return
    frame_duration = 1
    frame_length = int(frame_duration * sr)
    music = librosa.util.frame(music, frame_length=frame_length, hop_length=frame_length, axis=0)
    return music, sr


def download_data_from_google_drive(url):
    """
            Download datasets from google drive

            Args:
                url (str): The URL of the Google Drive folder.

            Returns:
                None
            """
    gdown.download_folder(url, quiet=True, use_cookies=False, remaining_ok=True)


def merge_datasets():
    """
        Merge music files and annotations into a single folder and file.

        The function renames the music files in the DEAM and PMEmo folders by adding a suffix to the filename,
        and then moves them to the 'all_music' folder. It also merges the annotations from DEAM and PMEmo into a
        single annotation file and saves it.

        Args:
            None

        Returns:
            None
        """
    # Merge music files into a single folser
    for song in os.listdir('../data/interim/deam/music'):
        os.rename(f'../data/interim/deam/music/{song}', f"../data/interim/deam/music/{song.split('.')[0]}_deam.wav")
    for song in os.listdir('../data/interim/PMEmo/music'):
        os.rename(f'../data/interim/PMEmo/music/{song}', f"../data/interim/PMEmo/music/{song.split('.')[0]}_pmemo.wav")
    for song in os.listdir('../data/interim/deam/music'):
        shutil.move(f'../data/interim/deam/music/{song}', '../data/interim/all_music')
    for song in os.listdir('../data/interim/PMEmo/music'):
        shutil.move(f'../data/interim/PMEmo/music/{song}', '../data/interim/all_music')

    # Merge annotation
    annotation_deam = pd.read_csv('../data/interim/deam/annotation.csv')
    annotation_pmemo = pd.read_csv('../data/interim/PMEmo/annotation.csv')
    annotation_deam['arousal'] = annotation_deam['arousal'].apply(lambda x: (((x + 1) * (1 - 0)) / (1 + 1)) + 0)
    annotation_deam['valence'] = annotation_deam['valence'].apply(lambda x: (((x + 1) * (1 - 0)) / (1 + 1)) + 0)
    annotation_deam['song_id'] = annotation_deam['song_id'].apply(lambda x: x + '_deam')
    annotation_pmemo['song_id'] = annotation_pmemo['song_id'].apply(lambda x: x + '_pmemo')
    annotation = pd.concat([annotation_deam, annotation_pmemo])
    annotation.reset_index(drop=True).to_csv('../data/interim/annotation.csv', index=False)
