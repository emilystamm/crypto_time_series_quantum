import csv
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
logger.setLevel(logging.INFO)

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
        
    def initialize_layers(self):
        logger.info("Initializing layers.")
        self.input_size, self.output_size = 6*(self.lookback- 1), 1
        n_layers = 1
        weight_shapes = {"weights": (n_layers, num_qubits, 3)}

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels=self.input_size, out_channels=128, kernel_size=3, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128,64,kernel_size=3, stride = 1, padding=1, bias=True)
        self.pool2 = nn.MaxPool1d(4)    
        self.fc1 = nn.Linear(64, 6)
        # self.fc1 = nn.Linear(64, 2)
        self.qlayer = qml.qnn.TorchLayer(qnode = self.circuit, weight_shapes = weight_shapes)
        self.fc2 = nn.Linear(6, 1)
        self.fc3 = nn.Linear(1, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        
    @qml.qnode(dev)
    def circuit(inputs, weights):
        AngleEmbedding(inputs, wires=range(num_qubits))
        StronglyEntanglingLayers(weights, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]

    def forward(self,x):
        logger.debug("Shape: {}".format(x.shape))
        in_size1 = x.size(0)  # one batch
        x = self.relu(self.conv1(x))
        logger.debug("Shape: {}".format(x.shape))
        x = self.pool1(x)
        logger.debug("Shape: {}".format(x.shape))
        x = self.relu(self.conv2(x))
        # print(x.shape)
        # x = self.pool2(x)
        logger.debug("Shape: {}".format(x.shape))
        x = x.view(in_size1, -1)  # flatten the tensor
        logger.debug("Shape: {}".format(x.shape))
        x = self.fc1(x)
        logger.debug("Shape: {}".format(x.shape))
        x = self.qlayer(x)
        logger.debug("Shape: {}".format(x.shape))
        x = self.fc2(x)
        logger.debug("Shape: {}".format(x.shape))
        output = self.fc3(x)
        logger.debug("Shape: {}".format(output.shape))
        return output


    def read(self, datafile, index_col, label_col):
        full_datafile = "../data/" + datafile
        logger.info("Reading in data from file {}".format(full_datafile))
        self.df = pd.read_csv(full_datafile, index_col = index_col, parse_dates=True)
        self.label_col = label_col


    def preprocess(self):
        logger.info("Preprocessing data.")

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
   
       
       
        X_train = X[self.start_index:self.start_index+self.num_train, :] 
        X_test = X[self.start_index+self.num_train:self.start_index +self.num_train + self.num_test, :]
        y_train = y[self.start_index:self.start_index+self.num_train, :]
        y_test = y[self.start_index+self.num_train:self.start_index+self.num_train + self.num_test, :]

        X_train  = Variable(Tensor(X_train))
        X_test = Variable(Tensor(X_test))

        self.X_train = reshape(X_train, (X_train.shape[0], X_train.shape[2] * X_test.shape[1], 1))
        self.X_test = reshape(X_test, (X_test.shape[0], X_test.shape[2] * X_test.shape[1], 1))

        y_train = Variable(Tensor(y_train))
        y_test = Variable(Tensor(y_test))

        print( y_train.shape)
        self.y_train = reshape(Variable(Tensor([y_train[i,5] for i in range(y_train.shape[0])])), (y_train.shape[0], 1))
        self.y_test = reshape(Variable(Tensor([y_test[i,5] for i in range(y_test.shape[0])])), (y_test.shape[0], 1))

        print(self.X_train.shape, X_train.shape)
        print(self.y_train.shape, y_train.shape)


    def train(self) -> None:
        logger.info("Begin training.")
        for epoch in range(self.epochs):
            batch_index = np.random.randint(0, self.y_train.shape[0], (self.batch_size))
            X_batch = self.X_train[batch_index]
            y_batch = self.y_train[batch_index]
            if epoch % 100 == 0: logger.info("Epoch {}".format(epoch))
            outputs = self.forward(X_batch)
            self.optimizer.zero_grad()
            logger.debug("X_batch.shape {} y_batch.shape {}".format(X_batch.shape, y_batch.shape))
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
        self.train_loss = loss.detach().numpy()
        logger.debug("Loss: {}".format(loss))

    def test(self) -> None:
        logger.info("Begin testing.")
        self.y_pred = self.forward(self.X_test)
        self.y_data_predict = self.y_pred.data.numpy() 
        self.y_data_test = self.y_test.data.numpy()
        self.test_loss = self.criterion(self.y_pred, self.y_test).detach().numpy()
        logger.info("Test Loss: {}".format(self.test_loss))
        logger.debug("Test {}\nPredict{}".format(self.y_data_test, self.y_data_predict))
        logger.debug("Test {}\nPredict{}".format(self.y_data_test.shape, self.y_data_predict.shape))

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
        plt.figure(figsize=(10,6)) 
        plt.plot(self.y_data_test, label='Actual Data')
        plt.plot(self.y_data_predict, label='Predicted Data') 
        plt.title('Crypto Time-Series Prediction\nLoss: {}'.format(round(float(self.test_loss),7)), fontsize=14)
        plt.savefig(full_plotfile)


model = CryptoTimeSeriesModel(
        num_train = 800, 
        num_test = 400,
        epochs = 5000,
        lr = .001,
        batch_size = 25,
        start_index = 100,
        lookback = 2
)

datafile = "ETH-USD.csv"
model.read(datafile, "Date", "Close")
model.preprocess()
model.initialize_layers()
model.train()
model.test()
model.write("test.csv")
model.plot("quantum_5_class_3.png")