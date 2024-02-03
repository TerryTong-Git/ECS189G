'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting(setting):
    dataset_test = None    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()
        loaded_test = self.dataset_test.load()

        X_train, y_train=loaded_data['X'], loaded_data['y']
        X_test, y_test=loaded_test['X'], loaded_test['y']


        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        score_list = self.evaluate.evaluate()
        # return np.mean(score_list), np.std(score_list)
        return 1,1 #random numbers

        