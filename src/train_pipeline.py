from src.data.make_deam_dataset import make_deam_dataset
from src.data.make_pmemo_dataset import make_pmemo_dataset
from src.features.get_mfcc import make_mfcc_dataset
from src.features.build_train_features import DataPreprocessor
import pandas as pd
import torch
import os
import shutil
from src.features.dataloader import MusicDataset
from models.model import EmotionModel
from src.models import train_model


def pipeline():
    make_deam_dataset('music_project/data/raw/DEAM/chorus',
                      'music_project/data/raw/DEAM/arousal.csv',
                      'music_project/data/raw/DEAM/valence.csv')

    make_pmemo_dataset('music_project/data/raw/PMEmo/chorus',
                       'music_project/data/raw/PMEmo/dynamic_features.csv')

    make_mfcc_dataset('music_project/data/interim/PMEmo/music',
                      'music_project/data/interim/PMEmo/annotation.csv',
                      'music_project/data/processed/pmemo')

    make_mfcc_dataset('music_project/data/interim/deam/music',
                      'music_project/data/interim/deam/annotation.csv',
                      'music_project/data/processed/deam')

    DataPreprocessor('/Users/tanchik/music_project/data/processed/deam/annotation.csv', 11800, 'deam')
    DataPreprocessor('/Users/tanchik/music_project/data/processed/pmemo/annotation.csv', 1850, 'pmemo')

    annotation_deam = pd.read_csv('music_project/data/finished/deam/annotation.csv')
    annotation_pmemo = pd.read_csv('music_project/data/finished/pmemo/annotation.csv')

    annotation_deam['arousal'] = annotation_deam['arousal'].apply(lambda x: (((x + 1) * (1 - 0)) / (1 + 1)) + 0)
    annotation_deam['valence'] = annotation_deam['valence'].apply(lambda x: (((x + 1) * (1 - 0)) / (1 + 1)) + 0)

    annotation_deam['song_id'] = annotation_deam['song_id'].apply(lambda x: x + '_deam')
    annotation_pmemo['song_id'] = annotation_pmemo['song_id'].apply(lambda x: x + '_pmemo')

    annotation = pd.concat([annotation_deam, annotation_pmemo])
    annotation.reset_index(drop=True).to_csv('music_project/data/finished/annotation.csv', index=False)
    x_train, x_valid = train_test_split(annotation, test_size=0.25, random_state=42)

    x_train.to_csv('music_project/data/finished/train_annotation.csv', index=False)
    x_valid.to_csv('music_project/data/finished/valid_annotation.csv', index=False)

    for song in os.listdir('music_project/data/finished/deam/mfcc'):
        os.rename(f'music_project/data/finished/deam/mfcc/{song}', f"music_project/data/finished/deam/mfcc/{song.split('.')[0]}_deam.npy")

    for song in os.listdir('music_project/data/finished/pmemo/mfcc'):
        os.rename(f'music_project/data/finished/pmemo/mfcc/{song}', f"music_project/data/finished/pmemo/mfcc/{song.split('.')[0]}_pmemo.npy")

    for song in os.listdir('music_project/data/finished/pmemo/mfcc'):
        shutil.move(f'music_project/data/finished/pmemo/mfcc/{song}', 'music_project/data/finished/deam/mfcc')

    train_dir = '/content/train_annotation.csv'
    valid_dir = '/content/valid_annotation.csv'
    mfcc_dir = '/content/mfcc_deam'
    train_dataset = MusicDataset(train_dir, mfcc_dir)
    valid_dataset = MusicDataset(valid_dir, mfcc_dir)

    train_size = len(train_dataset)
    valid_size = len(valid_dataset)

    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_data = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)

    model = EmotionModel()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    metric = nn.L1Loss()

    history = train_model(model, 600, optimizer, criterion, 'arousal', metric)
    history = train_model(model, 600, optimizer, criterion, 'valence', metric)

