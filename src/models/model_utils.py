from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import csv
from src.features.dataloader import MusicDataset


def split_annotation():
    df = pd.read_csv('../data/processed/annotation.csv')
    x_train, x_valid = train_test_split(df, test_size=0.25,
                                        random_state=42)
    x_train.to_csv('../data/processed/train_annotation.csv', index=False)
    x_valid.to_csv('../data/processed/valid_annotation.csv', index=False)


def get_tarin_valid_data():
    split_annotation()
    train_dir = '../data/processed/train_annotation.csv'
    valid_dir = '../data/processed/valid_annotation.csv'
    mfcc_dir = '../data/processed/mfcc'
    train_dataset = MusicDataset(train_dir, mfcc_dir)
    valid_dataset = MusicDataset(valid_dir, mfcc_dir)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_data = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    return train_data, valid_data


def make_history_file(directory):
    with open(f'{directory}/history.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows([['epoch', 'train_loss', 'train_metric', 'valid_loss', 'valid_metric']])

def print_results(epoch, history):
    print(f'epoch: {epoch}\n'
          f'train: loss {history["train_losses"][-1]:.4f}\n'
          f'train: metric {history["train_metrics"][-1]:.4f}\n'
          f'valid: loss {history["valid_losses"][-1]:.4f}\n'
          f'valid: metric {history["valid_metrics"][-1]:.4f}')
    print(f'{"-" * 35}')
    print()
