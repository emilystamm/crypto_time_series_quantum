import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
from torch import Tensor,  reshape
from torch.autograd import Variable 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoTimeSeriesModel(nn.Module):
    def __init__(self,
        num_train, num_test, epochs, lr, batch_size,start_index
    ) -> None:
        super(CryptoTimeSeriesModel, self).__init__()
        self.num_train = num_train
        self.num_test = num_test
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.start_index = start_index
        
    def initialize_layers(self, input_size, output_size):
        logger.info("Initializing layers.")
        self.input_size = input_size
        self.output_size = output_size
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(self.input_size, 128, kernel_size=3, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128,64,kernel_size=3, stride = 1, padding=1, bias=True)
        self.fc1 = nn.Linear(64, self.output_size)
        self.fc2 = nn.Linear(1, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        

    def read(self, datafile, index_col, label_col):
        full_datafile = "../data/" + datafile
        logger.info("Reading in data from file {}".format(full_datafile))
        self.df = pd.read_csv(full_datafile, index_col = index_col, parse_dates=True)
        self.y = self.df.loc[:, [label_col]]
        self.X = self.df.drop(label_col, 1)
        logger.debug("X = {}, y = {}".format(self.X,self.y))

    def preprocess(self):
        logger.info("Preprocessing data.")
        self.mm = MinMaxScaler() 
        self.ss = StandardScaler() 
        logger.info("Transform data.")
        X = self.ss.fit_transform(self.X)
        y = self.mm.fit_transform(self.y)
        logger.info("Split data into train and test. Training (testing) data size: {} ({}).".format(self.num_train, self.num_test))
        X_train = X[self.start_index:self.start_index+self.num_train, :] 
        X_test = X[self.start_index+self.num_train:self.start_index +self.num_train + self.num_test, :]
        y_train = y[self.start_index:self.start_index+self.num_train, :]
        y_test = y[self.start_index+self.num_train:self.start_index+self.num_train + self.num_test, :]

        X_train_tensor = Variable(Tensor(X_train))
        X_test_tensor = Variable(Tensor(X_test))

        self.X_train = reshape(X_train_tensor, (X_train_tensor.shape[0], X_train_tensor.shape[1], 1))
        self.X_test = reshape(X_test_tensor, (X_test_tensor.shape[0],  X_test_tensor.shape[1], 1))

        self.y_train = Variable(Tensor(y_train))
        self.y_test = Variable(Tensor(y_test))

    def forward(self,x):
        in_size1 = x.size(0)  # one batch
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = x.view(in_size1, -1)  # flatten the tensor
        x = self.fc1(x)
        output = self.fc2(x)
        return x

    def train(self) -> None:
        logger.info("Begin training.")
        for epoch in range(self.epochs):
            batch_index = np.random.randint(0, self.y_train.shape[0], (self.batch_size))
            X_batch = self.X_train[batch_index]
            y_batch = self.y_train[batch_index]
            if epoch % 100 == 0: logger.info("Epoch {}".format(epoch))
            outputs = self.forward(X_batch)
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
        self.train_loss = loss.detach().numpy()
        logger.debug("Loss: {}".format(loss))

    def test(self) -> None:
        logger.info("Begin testing.")
        self.y_pred = self.forward(self.X_test)#forward pass
        y_pred_np = self.y_pred.data.numpy() #numpy conversion
        y_test_np = self.y_test.data.numpy()
        self.y_data_predict = self.mm.inverse_transform(y_pred_np) #reverse transformation
        self.y_data_test = self.mm.inverse_transform(y_test_np)
        self.test_loss = self.criterion(self.y_pred, self.y_test).detach().numpy()
        logger.info("Test Loss: {}".format(self.test_loss))

    def write(self, writefile) -> None:
        full_writefile = "../results/" + writefile
        logger.info("Writing results to {}".format(full_writefile))
        with open(full_writefile, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            row = [self.num_train, self.num_test, self.batch_size, self.epochs, self.lr, self.train_loss, self.test_loss]
            writer.writerow(row)

    def plot(self, plotfile) -> None:
        full_plotfile= "../plots/" + plotfile
        logger.info("Plotting results and saving to {}".format(full_plotfile))
        plt.figure(figsize=(10,6)) #plotting
        plt.plot(self.y_data_test, label='Actual Data') #actual plot
        plt.plot(self.y_data_predict, label='Predicted Data') #predicted plot
        plt.title('Crypto Time-Series Prediction')
        plt.legend()
        plt.savefig(full_plotfile)


model = CryptoTimeSeriesModel(
        num_train = 800, 
        num_test = 400,
        epochs = 5000,
        lr = .001,
        batch_size = 25,
        start_index = 100
)
datafile = "ETH-USD.csv"
model.read(datafile, "Date", "Close")
model.preprocess()
model.initialize_layers(5, 1)
model.train()
model.test()
model.write("test.csv")
model.plot("result.png")