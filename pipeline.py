import shutil
import subprocess
from src.data.data_utils import download_folder_from_google_drive, download_file_from_doodle_drive
from src.data.make_pmemo_dataset import make_pmemo_dataset
from src.features.make_train_dataset import DataPreprocessor, make_mfcc_dataset
from src.models.train_model import Trainer
from src.models.predict_model import Predictor
from src.visualization.visualize import Visualizer


def pipeline():
    subprocess.call(['pip', 'install', '-r', './requirements.txt'])
    # Download raw data to the directory ./data/raw
    chorus_pmemo_url = 'https://drive.google.com/drive/folders/16uUDmtSzaWXfIZjIrnOGOw6DBd9d7jTO?usp=sharing'
    annotation_pmemo_url = 'https://drive.google.com/file/d/18NSlDIjoOCEp2JRsxMss9F1kbssj-hUL/view?usp=sharing'
    download_folder_from_google_drive(chorus_pmemo_url)
    download_file_from_doodle_drive(annotation_pmemo_url, 'data/raw/annotation.csv')
    shutil.move('chorus', 'data/raw')
    print('Pmemo raw dataset has been loaded')

    # Create a dataset with 1-second annotation and truncate music to one-second length
    pmemo_music_dir = 'data/raw/chorus'
    pmemo_annotation_dir = 'data/raw/annotation.csv'
    make_pmemo_dataset(pmemo_music_dir, pmemo_annotation_dir)
    print('Pmemo raw dataset has been preprocessed')

    # Create processed dataset with mfcc
    DataPreprocessor('data/interim/annotation.csv')
    make_mfcc_dataset('data/interim/music', 'data/processed/annotation.csv')
    print('MFCC (Mel Frequency Cepstral Coefficients) were extracted from the data')

    # Traning and prediction
    arousal_trainer = Trainer()
    arousal_trainer.train(4)
    valence_trainer = Trainer(target='valence')
    valence_trainer.train(4)
    predictor = Predictor()
    valence, arousal = predictor.predict('data/example/Alla_Pugachyova_Ya_tak_hochu_chtoby_leto_ne_konchalos.mp3')
    print(f'arousal vector: {arousal}')
    print(f'valence vector: {valence}')
    visualizer = Visualizer()
    visualizer.get_full_visualisation()


if __name__ == '__main__':
    pipeline()
