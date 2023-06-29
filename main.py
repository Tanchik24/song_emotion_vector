from starlette.staticfiles import StaticFiles

from src.models.predict_model import Predictor
import uvicorn
from fastapi import FastAPI, File, UploadFile
from client.app.app_utils import get_mean, get_std, get_qc
import pandas as pd
from io import BytesIO
import os
from dataclasses import dataclass
import random
from client.recomendation.song_recomendation import get_recommendation
from client.app.gpt import get_description
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
predictor = Predictor()

app.mount("/public", StaticFiles(directory="../public"), name="public")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class Song:
    author: str
    title: str


@app.post('/predict')
async def create_upload_file(song: UploadFile = File(...)):
    music = BytesIO(song.file.read())
    arousal, valence = predictor.predict(music)

    arousal = [float(val) for val in arousal]
    valence = [float(val) for val in valence]

    return {'arousal': arousal,
            'valence': valence}


@app.get('/predictValues')
async def get_values(filename: str):
    with open(filename, 'rb') as f:
        music = BytesIO(f.read())
        arousal, valence = predictor.predict(music)

        arousal = [float(val) for val in arousal]
        valence = [float(val) for val in valence]

        return {'arousal': arousal,
                'valence': valence}


@app.get('/mean')
async def get_result_mean(arousal: list, valence: list):
    valence, arousal = get_mean(valence, arousal)
    return {'arousal mean': round(float(arousal), 2),
            'valence mean ': round(float(valence), 2)}


@app.get('/std')
async def get_result_std(arousal: list, valence: list):
    valence, arousal = get_std(valence, arousal)
    return {'arousal std': round(float(arousal), 2),
            'valence std ': round(float(valence), 2)}


@app.get('/qc')
async def get_result_qc(arousal: list, valence: list):
    valence, arousal = get_qc(valence, arousal)
    return {'arousal qc': int(arousal),
            'valence qc ': int(valence)}


@app.get('/library')
async def get_library():
    df = pd.read_csv('songs.csv')
    tracks = list(df['id'].values)
    random_tracks = random.sample(tracks, 10)
    tracks_df = df[df['id'].isin(random_tracks)]
    artworks = os.listdir('client/public/artworks')
    random_artworks = random.sample(artworks, 10)
    tracks = [{'id': tracks_df.iloc[index]['id'], 'artist': tracks_df.iloc[index]['artist_name'],
               'title': tracks_df.iloc[index]['track_name'],
               'artwork': 'http://localhost:3001/public/artworks/' + random_artworks[index],
               'file': 'http://localhost:3001/public/music/' + tracks_df.iloc[index]['id'] + '.mp3'} for index in
              range(tracks_df.shape[0])]
    return tracks


@app.get('/recommendation')
async def get_songs_recommendation(ids):
    recommendation = get_recommendation(ids)
    return recommendation


@app.get('/gpt_description')
async def get_gpt_description(arousal: list, valence: list):
    openai_key = 'key'
    content = get_description(arousal, valence, openai_key)


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8001)
