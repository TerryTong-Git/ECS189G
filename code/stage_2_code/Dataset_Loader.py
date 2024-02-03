'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pandas as pd

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        df = pd.read_csv(self.dataset_source_folder_path + self.dataset_source_file_name)
        y = df.iloc[:, 0].fillna(0)
        X = df.iloc[:, 1:].fillna(0)
        y = y.to_numpy()
        X = X.to_numpy()
        return {'X': X, 'y': y}