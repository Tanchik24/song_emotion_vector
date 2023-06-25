import librosa
import os
import shutil
import pandas as pd
import random


class DataPreprocessor():
  """
    Create annotation for balanced dataset
    """
  def __init__(self, df_dir, size, name):
    self.__df = pd.read_csv(df_dir)
    self.__make_new_dataset(size=size)


  def __get_quadrants(self, arousal, valence):
    """
    Determine the quadrant based on arousal and valence values.

    Args:
      arousal (float): Arousal value.
      valence (float): Valence value.

    Returns:
      int: The quadrant number (1, 2, 3, or 4).
    """
    mid_val = 0.5
    if valence > mid_val:
      return 4 if arousal < mid_val else 1
    else:
      return 3 if arousal < mid_val else 2


  def __add_quadrants_columns(self, quadrants):
    """
    Add a 'quadrant' column to the DataFrame.

    Args:
      quadrants (list): List of quadrant numbers.

    Returns:
      pandas.DataFrame: Updated DataFrame with the 'quadrant' column.
    """
    qudrants_df = pd.DataFrame(quadrants, columns = ['quadrant'])
    self.__df = pd.concat([self.__df, qudrants_df], axis=1)
    return self.__df



  def __count_quadrants(self):
    """
    Count the number of samples in each quadrant.

    Returns:
      tuple: A tuple containing a list of quadrants and a Counter object with quadrant counts.
    """
    quadrants = []
    for row in range(self.__df.shape[0]):
      arousal = self.__df.iloc[row]['arousal']
      valence = self.__df.iloc[row]['valence']
      quadrants.append(self.__get_quadrants(arousal, valence))
    quadrants_counter = Counter(quadrants)
    for key in quadrants_counter.keys():
      print(f'Quadrat {key} : {quadrants_counter[key]} samples')
    return quadrants, quadrants_counter


  def __save_csv(self):
    """
    Save the DataFrame as a CSV file.

    Returns:
      None
    """
    self.__df.to_csv(f'./data/finished/{name}temp_annotation.csv', index=False)


  def __make_new_dataset(self, size=1500):
    """
    Create a new dataset with a specified size by balancing the samples in each quadrant.

    Args:
      size (int): The desired size of each quadrants.

    Returns:
      None
    """
    quadrants, quadrants_counter = self.__count_quadrants()
    self.__df = self.__add_quadrants_columns(quadrants)
    for key in quadrants_counter.keys():
      quadrant_size = quadrants_counter[key]
      indexes = list(self.__df[self.__df['quadrant'] == key].index)

      if quadrant_size >= size:
        del_size = quadrants_counter[key] - size
        del_indexes = random.sample(indexes, del_size)
        self.__df.drop(index=del_indexes, inplace=True)

      else:
        added_size = size - quadrant_size
        added_indexes =  random.sample(indexes, added_size)
        self.__df = pd.concat([self.__df, self.__df.loc[added_indexes]], axis=0).reset_index(drop=True)
    self.__save_csv()
