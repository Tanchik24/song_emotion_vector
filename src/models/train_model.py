import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import json
import csv
from models.model import EmotionModel
from src.models.model_utils import get_tarin_valid_data, make_history_file, print_results


class Trainer():
    def __init__(self, target='arousal'):
        self.model = EmotionModel()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.target = target
        self.directory = self.__get_directory()
        self.history = {'train_losses': [], 'valid_losses': [],
                        'train_metrics': [], 'valid_metrics': []}
        self.best_metrics = {'epoch': 0, 'valid_loss': 0, 'valid_metric': 0}
        self.train_data, self.valid_data = get_tarin_valid_data()
        self.epoch_counter = 0
        self.criterion = nn.MSELoss()
        self.metric = nn.L1Loss()
        make_history_file(self.directory)

    def __save_weighst(self, epoch):
        torch.save(self.model.state_dict(), f'{self.directory}/each_epochs.pt')
        if epoch == 100:
            torch.save(self.model.state_dict(), f'{self.directory}/100_epochs.pt')
        if epoch == 199:
            torch.save(self.model.state_dict(), f'{self.directory}/200_epochs.pt')

    def __add_best_results(self, epoch):
        self.best_metrics['epoch'] = epoch
        self.best_metrics['valid_loss'] = self.history["valid_losses"][-1]
        self.best_metrics['valid_metric'] = self.history["valid_metrics"][-1]
        torch.save(self.model.state_dict(), f'{self.directory}/best_model.pt')
        with open(f'{self.directory}/best_res.json', 'w') as outfile:
            json.dump(self.best_metrics, outfile)
        print(
            f"best results: epoch {self.best_metrics['epoch']}, valid loss {self.best_metrics['valid_loss']}, valid metric {self.best_metrics['valid_metric']}")

    def __add_to_csv_file(self, epoch):
        with open(f'{self.directory}/history.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerows([[epoch, round(self.history["train_losses"][-1], 4),
                               round(self.history["train_metrics"][-1], 4),
                               round(self.history["valid_losses"][-1], 4),
                               round(self.history["valid_metrics"][-1], 4)]])

    def __get_directory(self):
        directory = '../models/train_weights/arousal' if self.target == 'arousal' else '../models/train_weights/valence'
        return directory

    def __tarin_template(self, optimizer=None):

        losses_iter = []
        metrics_iter = []
        data = self.train_data if optimizer is not None else self.valid_data

        for music, arousal, valence in data:
            music, arousal, valence = music.to(self.device), arousal.to(self.device), valence.to(self.device)
            out = self.model(music)

            if self.target == 'arousal':
                loss = torch.sqrt(self.criterion(out.squeeze(), arousal))
                metric_res = self.metric(out.squeeze(), arousal)
            elif self.target == 'valence':
                loss = torch.sqrt(self.criterion(out.squeeze(), valence))
                metric_res = self.metric(out.squeeze(), valence)

            losses_iter.append(loss.item())
            metrics_iter.append(metric_res.item())

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return np.mean(losses_iter), np.mean(metrics_iter)

    def train(self, epochs, keep=False):

        if keep is True:
            try:
                self.model.load_state_dict(torch.load(f'{self.directory}/each_epoch', map_location=self.device))
            except FileNotFoundError:
                print(f'Файлов с весами не обнаружено')
                return

        optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.01)

        for epoch in range(self.epoch_counter, epochs):
            self.model.train()
            self.epoch_counter += 1

            train_losses_iter, train_metrics_iter = self.__tarin_template(optimizer=optimizer)
            self.history['train_losses'].append(np.mean(train_losses_iter))
            self.history['train_metrics'].append(np.mean(train_metrics_iter))

            self.model.eval()
            valid_losses_iter, valid_metrics_iter = self.__tarin_template(optimizer=None)
            self.history['valid_losses'].append(np.mean(valid_losses_iter))
            self.history['valid_metrics'].append(np.mean(valid_metrics_iter))

            self.__add_to_csv_file(epoch)

            if (self.best_metrics['valid_loss'] == 0) or (
                    self.best_metrics['valid_loss'] > self.history["valid_losses"][-1]):
                self.__add_best_results(epoch)
            self.__save_weighst(epoch)
            print_results(epoch, self.history)
        print(f'Обучение закончено')
