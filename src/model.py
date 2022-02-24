import csv
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import QNode

import torch 
import torch.nn as nn
from torch import Tensor,  reshape
from torch.autograd import Variable 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

global num_qubits
num_qubits = 6

class CryptoTimeSeriesModel(nn.Module):
    '''
    Init
    '''
    def __init__(self,
        num_train, num_test, epochs, lr, batch_size,start_index, lookback
    ) -> None:
        logger.info("Initialize model.")
        super(CryptoTimeSeriesModel, self).__init__()
        # Initialize 
        self.num_train = num_train
        self.num_test = num_test
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.start_index = start_index
        self.lookback= lookback

    '''
    Read
    read data from file
    '''
    def read(self, datafile, index_col):
        full_datafile = "../data/" + datafile
        logger.info("Reading in data from file {}".format(full_datafile))
        self.df = pd.read_csv(full_datafile, index_col = index_col, parse_dates=True)

    '''
    Preprocess
    format as sequences, split into train and test, and transformation of datapoints 
    '''
    def preprocess(self, y_col):
        logger.info("Preprocessing data.")
        self.y_col = y_col

        # Reshape data into sequences
        # Old shape: (number of datapoints, number of columns)
        # New shape: (len(raw_data)-lookback), lookback - 1, number of columns)
        raw_data = self.df.to_numpy() 
        data = []
        for index in range(len(raw_data) - self.lookback): 
            data.append(raw_data[index: index + self.lookback])
        
        # Split into X and y 
        # X shape: (number of datapoints, lookback - 1, number of columns)
        # y shape: (number of datapoints, number of columns)
        X = np.array(data)[:, :-1, :]
        y = np.array(data)[:, -1, :]
        logger.debug("X shape {}, y shape {}".format(X.shape, y.shape))

        # Data transformation
        logger.info("Transform data.")
        self.mm = MinMaxScaler() 
        self.ss = StandardScaler() 
        y = self.mm.fit_transform(y)
  
        # Split into train and test
        logger.info("Split data into train and test. Training (testing) data size: {} ({}).".format(self.num_train, self.num_test))       
        self.X_train_1 = X[self.start_index:self.start_index+self.num_train, :] 
        self.X_test_1 = X[self.start_index+self.num_train:self.start_index +self.num_train + self.num_test, :]
        self.y_train_1 = y[self.start_index:self.start_index+self.num_train, :]
        self.y_test_1 = y[self.start_index+self.num_train:self.start_index+self.num_train + self.num_test, :]

        # Various formats/shapes 
        X_train_2  = Variable(Tensor(self.X_train_1))
        X_test_2 = Variable(Tensor(self.X_test_1))
        y_train_2 = Variable(Tensor(self.y_train_1))
        y_test_2 = Variable(Tensor(self.y_test_1))
        
        # Used for CNN (+ Q)
        self.X_train = reshape(X_train_2, (X_train_2.shape[0], X_train_2.shape[2] * X_train_2.shape[1], 1))
        self.X_test = reshape(X_test_2, (X_test_2.shape[0], X_test_2.shape[2] * X_test_2.shape[1], 1))
        self.y_train = reshape(Variable(Tensor([y_train_2[i,self.y_col] for i in range(y_train_2.shape[0])])), (y_train_2.shape[0], 1))
        self.y_test = reshape(Variable(Tensor([y_test_2[i,self.y_col] for i in range(y_test_2.shape[0])])), (y_test_2.shape[0], 1))

        # Used for q sequential model 
        self.y_train_1 = reshape(Variable(Tensor([self.y_train[i,self.y_col] for i in range(self.y_train.shape[0])])), (self.y_train.shape[0],))
        self.y_test_1 = reshape(Variable(Tensor([self.y_test[i,self.y_col] for i in range(self.y_test.shape[0])])), (self.y_test.shape[0],))
     
    '''
    Write
    write loss, timings, and some parameter choices to file  
    '''
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
    '''
    Plot
    '''
    def plot(self, plotfile) -> None:
        full_plotfile= "../plots/" + plotfile
        logger.info("Plotting results and saving to {}".format(full_plotfile))
        plt.figure(figsize=(10,6)) 
        plt.plot(self.y_data_test, label='Actual Data')
        plt.plot(self.y_data_predict, label='Predicted Data') 
        plt.title('Crypto Time-Series Prediction\nLoss: {}'.format(round(float(self.test_loss),7)), fontsize=14)
        plt.savefig(full_plotfile)


