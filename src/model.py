import csv
import time 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import QNode
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers

import torch 
import torch.nn as nn
from torch import Tensor,  reshape
from torch.autograd import Variable 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

num_qubits = 6
dev = qml.device("default.qubit.torch", wires=num_qubits)


class CryptoTimeSeriesModel(nn.Module):
    def __init__(self,
        num_train, num_test, epochs, lr, batch_size,start_index, lookback
    ) -> None:
        logger.info("Initialize model.")
        super(CryptoTimeSeriesModel, self).__init__()
        self.num_train = num_train
        self.num_test = num_test
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.start_index = start_index
        self.lookback= lookback

    def read(self, datafile, index_col):
        full_datafile = "../data/" + datafile
        logger.info("Reading in data from file {}".format(full_datafile))
        self.df = pd.read_csv(full_datafile, index_col = index_col, parse_dates=True)

    def preprocess(self, y_col):
        logger.info("Preprocessing data.")
        self.y_col = y_col

        data_raw = self.df.to_numpy() # convert to numpy array
        data = []
        
        # create all possible sequences of length seq_len
        for index in range(len(data_raw) - self.lookback): 
            data.append(data_raw[index: index + self.lookback])
        
        self.df = data

        self.mm = MinMaxScaler() 
        self.ss = StandardScaler() 
        logger.info("Transform data.")

        logger.info("Split data into train and test. Training (testing) data size: {} ({}).".format(self.num_train, self.num_test))
       
        X = np.array(data)[:, :-1, :]
        y = np.array(data)[:, -1, :]

        y = self.mm.fit_transform(y)
  
    
        self.X_train_orig = X[self.start_index:self.start_index+self.num_train, :] 
        self.X_test_orig = X[self.start_index+self.num_train:self.start_index +self.num_train + self.num_test, :]
        self.y_train_orig = y[self.start_index:self.start_index+self.num_train, :]
        self.y_test_orig = y[self.start_index+self.num_train:self.start_index+self.num_train + self.num_test, :]

        X_train  = Variable(Tensor(self.X_train_orig ))
        X_test = Variable(Tensor(self.X_test_orig))

        self.X_train = reshape(X_train, (X_train.shape[0], X_train.shape[2] * X_train.shape[1], 1))
        self.X_test = reshape(X_test, (X_test.shape[0], X_test.shape[2] * X_test.shape[1], 1))
        
        y_train = Variable(Tensor(self.y_train_orig))
        y_test = Variable(Tensor(self.y_test_orig))

        self.y_train = reshape(Variable(Tensor([y_train[i,self.y_col] for i in range(y_train.shape[0])])), (y_train.shape[0], 1))
        self.y_test = reshape(Variable(Tensor([y_test[i,self.y_col] for i in range(y_test.shape[0])])), (y_test.shape[0], 1))

        self.y_train_orig = reshape(Variable(Tensor([self.y_train[i,self.y_col] for i in range(self.y_train.shape[0])])), (self.y_train.shape[0],))
        self.y_test_orig = reshape(Variable(Tensor([self.y_test[i,self.y_col] for i in range(self.y_test.shape[0])])), (self.y_test.shape[0],))
     
        logger.debug("x_train[0] {} y_train[0] {}".format(self.X_train[0], self.y_train[0]))

    def write(self, writefile) -> None:
        full_writefile = "../results/" + writefile
        logger.info("Writing results to {}".format(full_writefile))
        with open(full_writefile, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            row = [
                self.type, self.y_col,
                round(self.train_time,3), round(self.test_time,3), self.num_train, self.num_test, 
                self.start_index,
                self.batch_size, self.epochs, self.lr, self.train_loss, self.test_loss
            ]
            writer.writerow(row)

    def plot(self, plotfile) -> None:
        full_plotfile= "../plots/" + plotfile
        logger.info("Plotting results and saving to {}".format(full_plotfile))
        plt.figure(figsize=(10,6)) 
        plt.plot(self.y_data_test, label='Actual Data')
        plt.plot(self.y_data_predict, label='Predicted Data') 
        plt.title('Crypto Time-Series Prediction\nLoss: {}'.format(round(float(self.test_loss),7)), fontsize=14)
        plt.savefig(full_plotfile)


