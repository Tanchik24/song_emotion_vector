import pandas as pd
import random
import os
import subprocess

HOST = "http://127.0.0.1:8005"


def get_library():
    print(subprocess.call(['pwd']))
    df = pd.read_csv('webapi/songs.csv')
    tracks = list(df['id'].values)
    random_tracks = random.sample(tracks, 10)
    tracks_df = df[df['id'].isin(random_tracks)]
    artworks = os.listdir('public/artworks')
    random_artworks = random.sample(artworks, 10)
    tracks = [{'id': tracks_df.iloc[index]['id'], 'artist': tracks_df.iloc[index]['artist_name'],
               'title': tracks_df.iloc[index]['track_name'],
               'artwork': f'{HOST}/public/artworks/' + random_artworks[index],
               'file': f'{HOST}/public/music/' + tracks_df.iloc[index]['id'] + '.mp3'} for index in
              range(tracks_df.shape[0])]
    return tracks
