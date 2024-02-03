'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import wandb

class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 1000
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        self.layer1 = nn.Linear(784,500)
        self.layer2 = nn.Linear(500,500)
        self.layer3 = nn.Linear(500,500)
        self.layer4 = nn.Linear(500,500)
        self.layer5 = nn.Linear(500,500)
        self.layer6 = nn.Linear(500,10)
        self.activation = nn.ReLU()


        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # layerlist = []
        # layerlist.append(nn.Linear(784, 300))  # n_in input neurons connected to i number of output neurons
        # layerlist.append(nn.ReLU(inplace=True))  # Apply activation function - ReLU
        # layerlist.append(nn.BatchNorm1d(300))  # Apply batch normalization
        # layerlist.append(nn.Dropout(p = 0.5))  # Apply dropout to prevent overfitting
        # nn.init.kaiming_normal_(layerlist[-4].weight)
        # n_in = 300
        # out_sz = 10
        # for i in range(10):
        #     layerlist.append(nn.Linear(300, 300))  # n_in input neurons connected to i number of output neurons
        #     layerlist.append(nn.ReLU(inplace=True))  # Apply activation function - ReLU
        #     layerlist.append(nn.BatchNorm1d(300))  # Apply batch normalization
        #     layerlist.append(nn.Dropout(p = 0.5))  # Apply dropout to prevent overfitting
        #     nn.init.kaiming_normal_(layerlist[-4].weight, nonlinearity="relu")
        # layerlist.append(nn.Linear(300, 10))
        # nn.init.kaiming_normal_(layerlist[-1].weight, nonlinearity="relu")

        # self.layers = nn.Sequential(*layerlist)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
    
        self.activation_func = nn.Softmax(dim=1)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        h = self.activation(self.layer1(x))
        h = self.activation(self.layer2(h))
        h = self.activation(self.layer3(h))
        h = self.activation(self.layer4(h))
        h = self.activation(self.layer5(h))
       

        # hidden layer embeddings
        # h = x
        # for layer in self.layers:
        #     a = layer(h)
        #     if h.size() == a.size():
        #         h = a + h  # Add the residual connection
        #     else:
        #         h = a
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func(self.layer6(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html betas=(0.9, 0.999) 
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate) #use betas for momentum
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y))
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch%100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                evaluated = accuracy_evaluator.evaluate()
                wandb.log({"acc": evaluated[3], "loss": train_loss.item()})
                print('Epoch:', epoch,'Accuracy', evaluated[3], 'Loss:', train_loss.item())
                print('F1 micro:', evaluated[0][0], 'F1 macro:', evaluated[0][1], 'F1 weight:', evaluated[0][2])
                print('recall micro:', evaluated[1][0], 'recall macro:', evaluated[1][1], 'recall weight:', evaluated[1][2])
                print('precision micro:', evaluated[2][0], 'precision macro:', evaluated[2][1], 'precision weight:', evaluated[2][2])
    
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        print(self.data['train'])
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
