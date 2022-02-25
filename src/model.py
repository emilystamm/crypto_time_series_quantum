"""
===================================================
MODEL.PY
CryptoTimeSeries base model; inherits from nn.Module
===================================================
"""
# Imports 
from asyncore import write
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
from torch import Tensor, reshape
from torch.autograd import Variable 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set number of qubits, in this case equal to the 6 columns for angle embedding
global num_qubits
num_qubits = 6

import string
import random
'''
ID_Generator
Helper function to create unique id for each run
'''
def id_generator(size=4, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

"""
CryptoTimeSeriesModel
Base model for reading and preprocessing data, and writing and plotting results.
"""
class CryptoTimeSeriesModel(nn.Module):
    '''
    Init
    '''
    def __init__(self,
        num_train, num_test, iterations, lr, batch_size,start_index, lookback,
        quantum=None, conv=None, num_layers = None
    ) -> None:
        logger.info("Initialize model.")
        super(CryptoTimeSeriesModel, self).__init__()
        # Initialize 
        self.num_train = num_train
        self.num_test = num_test
        self.iterations = iterations
        self.lr = lr
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.start_index = start_index
        self.lookback= lookback
        self.quantum = quantum
        self.conv = conv
        self.num_layers = num_layers
        self.weights = None 
        self.bias = None
        self.id = id_generator()

    '''
    Read
    read data from file
    '''
    def read(self, datafile, index_col):
        full_datafile = "../data/" + datafile
        logger.info("Reading in data from file {}".format(full_datafile))
        self.df = pd.read_csv(full_datafile, index_col = index_col, parse_dates=True)
        self.dates = pd.read_csv(full_datafile)["Date"][self.start_index + self.num_train: self.start_index + self.num_train + self.num_test]


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
        y = np.array([elt[y_col] for elt in y]).reshape(y.shape[0], 1)
        logger.debug("X shape {}, y shape {}".format(X.shape, y.shape))

        # Data transformation
        logger.info("Transform data.")
        self.mm = MinMaxScaler() 
        self.ss = StandardScaler() 
        X = np.array([self.ss.fit_transform(x) for x in X])
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
        self.X_train = reshape(X_train_2, (X_train_2.shape[0],1, X_train_2.shape[2] * X_train_2.shape[1]))
        self.X_test = reshape(X_test_2, (X_test_2.shape[0], 1, X_test_2.shape[2] * X_test_2.shape[1]))
        self.y_train = y_train_2
        self.y_test = y_test_2
        
        # Used for q sequential model 
        self.y_train_1 = reshape(Variable(Tensor(self.y_train)), (self.y_train.shape[0],1))
        self.y_test_1 = reshape(Variable(Tensor(self.y_test)), (self.y_test.shape[0],1))

        # writefile = "preprocess.csv"
        # # np.savetxt("../results/X_train_{}_{}".format(self.type, writefile), self.X_train_1, delimiter=",")
        # # np.savetxt("../results/y_train_{}_{}".format(self.type, writefile), self.y_train_1, delimiter=",")
        # # np.savetxt("../results/X_test_{}_{}".format(self.type, writefile), self.X_test_1, delimiter=",")
        # # np.savetxt("../results/y_test_{}_{}".format(self.type, writefile), self.y_test_1, delimiter=",")
        # # input()

        

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
                self.id, self.type, self.y_col,
                round(self.train_time,3), round(self.test_time,3), self.num_train, self.num_test, 
                self.start_index,
                self.batch_size, self.iterations, self.lr, self.train_loss, self.test_loss,
                self.num_layers, self.conv, self.quantum, self.weights, self.bias
            ]
            writer.writerow(row)
        np.savetxt("../results/y/y_test_{}_{}_{}".format(self.type, self.id, writefile), self.invtransformed_y_test, delimiter=",")
        np.savetxt("../results/y/y_predict_{}_{}_{}".format(self.type,self.id, writefile), self.invtransformed_y_predict, delimiter=",")

    '''
    Inverse transform of y 
    '''
    def invtransform_y(self):
        self.invtransformed_y_test = self.mm.inverse_transform(
            reshape(Variable(Tensor(self.y_data_test)),  (self.y_data_test.shape[0], 1)))
        self.invtransformed_y_predict = self.mm.inverse_transform(
            reshape(Variable(Tensor(self.y_data_predict)),  (self.y_data_predict.shape[0], 1)))
        return self.invtransformed_y_test, self.invtransformed_y_predict
        
    '''
    Plot
    '''
    def plot(self, plotfile, y_preds = {}) -> None:
        full_plotfile= "../plots/" + self.id + "_" + plotfile
        logger.info("Plotting results and saving to {}".format(full_plotfile))
        fig, (ax1) = plt.subplots(1,1, figsize=(10,6))
        display_dates = np.array(self.dates)
        ax1.plot(display_dates, self.invtransformed_y_test, label='Actual Data')
        ax1.plot(display_dates, self.invtransformed_y_predict, label='{} Predicted Data'.format(self.type))
        for y_pred_type in y_preds.keys():
            ax1.plot(display_dates, y_preds[y_pred_type], label='{} Predicted Data'.format(y_pred_type))
        ax1.legend()
        ax1.set_xticks(display_dates[::8])
        ax1.set_xticklabels(display_dates[::8], rotation=30)
        ax1.set_title('Crypto Time-Series Prediction\nLoss: {}'.format(round(float(self.test_loss),7)), fontsize=14)
        plt.savefig(full_plotfile)



