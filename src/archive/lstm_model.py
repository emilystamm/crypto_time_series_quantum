import time 
import numpy as np
import pennylane as qml

import torch 
import torch.nn as nn
from torch import Tensor,  reshape
from torch.autograd import Variable 
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from model import CryptoTimeSeriesModel, num_qubits

# num_qubits = 6
# dev = qml.device("default.qubit.torch", wires=num_qubits)
dev = qml.device("default.qubit", wires=num_qubits)


class LSTMCryptoTimeSeriesModel(CryptoTimeSeriesModel):
    def __init__(self,
        num_train, num_test, epochs, lr, batch_size,start_index, lookback
    ) -> None:
        super(LSTMCryptoTimeSeriesModel, self).__init__(num_train, num_test, epochs, lr, batch_size,start_index, lookback)        
        self.num_layers = 2
        self.input_size = 1
        self.hidden_size = 16
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc_1 = nn.Linear(self.hidden_size, 128)
        self.fc = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
    
    def forward(self, X):
        h_0 = Variable(torch.zeros(self.num_layers, X.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, X.size(0), self.hidden_size))
        output, (h_n, c_n) = self.lstm(X, (h_0, c_0))
        h_n = h_n.view(-1, self.hidden_size)
        out = self.relu(h_n)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out


    def train(self) -> None:
        logger.info("Begin training.")
        for epoch in range(self.epochs):
            batch_index = np.random.randint(0, len(self.y_train), (self.batch_size))
            X_batch = self.X_train[batch_index]
            y_batch = self.y_train[batch_index]
            if epoch % 10 == 0: logger.info("Epoch {}".format(epoch))
            outputs = self.forward(X_batch)
            self.optimizer.zero_grad()
            logger.debug("outputs\n{}\ny_batch\n{}".format(outputs.shape, y_batch.shape))
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

    def test(self) -> None:
        logger.info("Begin testing.")
        train_predict = self.forward(self.df_X_ss)#forward pass
        data_predict = train_predict.data.numpy() #numpy conversion
        dataY_plot = self.df_y_mm.data.numpy()
        self.data_predict = self.mm.inverse_transform(data_predict) #reverse transformation
        self.dataY_plot = self.mm.inverse_transform(dataY_plot)
  
