import shutil
from src.data.data_utils import download_data_from_google_drive, merge_datasets
from src.data.make_pmemo_dataset import make_pmemo_dataset
from src.data.make_deam_dataset import make_deam_dataset
from src.features.make_train_dataset import DataPreprocessor, make_mfcc_dataset
from src.models.train_model import Trainer
from src.models.predict_model import Predictor
from src.visualization.visualize import Visualizer

def pipeline():
    # Download raw data to the directory ./data/raw
    pmemo_url = 'https://drive.google.com/drive/folders/1gH171hJVZMBnXaMF-ikfUt1Ngrhz4ufp?usp=sharing'
    download_data_from_google_drive(pmemo_url)
    shutil.move('PMEmo2019', '../../data/raw')
    print('Pmemo raw dataset has been loaded')
    deam_url = 'https://drive.google.com/drive/folders/1aFxmM_DSA_2cMFe-RdtWhBCXLEHKFHQy?usp=drive_link'
    download_data_from_google_drive(deam_url)
    shutil.move('deam', '../../data/raw')
    print('Deam raw dataset has been loaded')

    # Create a dataset with 1-second annotation and truncate music to one-second length
    pmemo_music_dir = '../../data/raw/PMEmo2019/chorus'
    pmemo_annotation_dir = '../../data/raw/PMEmo2019/dynamic_annotations.csv'
    make_pmemo_dataset(pmemo_music_dir, pmemo_annotation_dir)
    print('Pmemo raw dataset has been preprocessed')
    deam_music_dir = '../../data/raw/deam/music'
    deam_annotation_arousal_dir = '../../data/raw/deam/arousal.csv'
    deam_annotation_valence_dir = '../../data/raw/deam/valence.csv'
    make_deam_dataset(deam_music_dir, deam_annotation_arousal_dir, deam_annotation_valence_dir)
    print('Deam raw dataset has been preprocessed')

    # Create processed dataset with mfcc
    merge_datasets()
    print('Datasets were merged')
    DataPreprocessor('../../data/interim/annotation.csv')
    make_mfcc_dataset('../../data/interim/all_music', '../data/processed/annotation.csv')
    print('MFCC (Mel Frequency Cepstral Coefficients) were extracted from the data')

    # Traning and prediction
    arousal_trainer = Trainer()
    arousal_trainer.train(4)
    valence_trainer = Trainer(target='valence')
    valence_trainer.train(4)
    predictor = Predictor()
    valence, arousal = predictor.predict('../data/example/Alla_Pugachyova_Ya_tak_hochu_chtoby_leto_ne_konchalos.mp3')
    print(f'arousal vector: {arousal}')
    print(f'valence vector: {valence}')
    visualizer = Visualizer()
    visualizer.get_full_visualisation()


if __name__ == '__main__':
    pipeline()