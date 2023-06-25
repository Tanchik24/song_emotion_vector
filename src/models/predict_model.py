import torch
import soundfile as sf
import librosa

from models.model import EmotionModel
from src.features.features_utils import make_mono_sound, cut_music, preprocess_song

class Predictor:
    def __init__(self):
        self.model_arousal = EmotionModel()
        self.model_valence = EmotionModel()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_arousal.load_state_dict(torch.load('/Users/tanchik/music_project/models/best_model_arousal.pt', map_location=self.device))
        self.model_valence.load_state_dict(torch.load('/Users/tanchik/music_project/models/best_model_valence.pt', map_location=self.device))
        self.model_arousal, self.model_valence = self.model_arousal.to(self.device), self.model_valence.to(self.device)
        self.model_arousal.eval()
        self.model_valence.eval()


    def predict(self, song):
        data, sr = sf.read(song)
        data = make_mono_sound(data)
        data = cut_music(data, sr)

        arousal = []
        valence = []
        for elem in data:
            mfcc = preprocess_song(elem, sr)
            valence_sec = self.model_valence(mfcc.to(self.device)).squeeze()
            arousal_sec = self.model_arousal(mfcc.to(self.device)).squeeze()
            valence.append(valence_sec.cpu().detach().numpy())
            arousal.append(arousal_sec.cpu().detach().numpy())
        return valence, arousal
