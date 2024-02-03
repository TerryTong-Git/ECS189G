'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        f1 = self.eval_f1()
        recall = self.eval_recall()
        precision = self.eval_precision()
        accuracy = accuracy_score(self.data['true_y'], self.data['pred_y'])
        return [f1,recall,precision,accuracy]
    def eval_f1(self):
        macro =  f1_score(self.data['true_y'], self.data['pred_y'], average='macro')
        micro =  f1_score(self.data['true_y'], self.data['pred_y'], average='micro')
        weighted =  f1_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        return [micro,macro,weighted]
    def eval_recall(self):
        macro =  recall_score(self.data['true_y'], self.data['pred_y'], average='macro')
        micro =  recall_score(self.data['true_y'], self.data['pred_y'], average='micro')
        weighted =  recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        return [micro,macro,weighted]
    def eval_precision(self):
        macro =  precision_score(self.data['true_y'], self.data['pred_y'], average='macro')
        micro =  precision_score(self.data['true_y'], self.data['pred_y'], average='micro')
        weighted =  precision_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        return [micro,macro,weighted]    
        