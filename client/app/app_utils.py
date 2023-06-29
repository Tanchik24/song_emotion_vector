import numpy as np

def get_mean(valence, arousal):
    return np.mean(valence), np.mean(arousal)

def get_std(valence, arousal):
    return np.std(valence), np.std(arousal)

def get_qc(valence, arousal):
    arousal_qc = sum(1 for i in range(1, len(arousal)) if arousal[i] * arousal[i-1] < 0)
    valence_qc = sum(1 for i in range(1, len(valence)) if valence[i] * valence[i-1] < 0)
    return valence_qc, arousal_qc