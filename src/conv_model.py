import time 
import numpy as np
import pennylane as qml

import torch 
import torch.nn as nn

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from model import CryptoTimeSeriesModel

num_qubits = 6
dev = qml.device("default.qubit.torch", wires=num_qubits)


class ConvCryptoTimeSeriesModel(CryptoTimeSeriesModel):
    def __init__(self,
        num_train, num_test, epochs, lr, batch_size,start_index, lookback
    ) -> None:
        super(ConvCryptoTimeSeriesModel, self).__init__(num_train, num_test, epochs, lr, batch_size,start_index, lookback)
        
    def initialize_layers(self, conv = None, quantum = None):
        logger.info("Initializing layers.")
        if quantum: self.type =  "CNN+Quantum"
        else: self.type = "CNN"
        self.input_size, self.output_size = 6*(self.lookback- 1), 1
        n_layers = 1
        weight_shapes = {"weights": (n_layers, num_qubits, 3)}
        self.conv = conv
        self.quantum = quantum
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels=self.input_size, out_channels=conv[0], kernel_size=3, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(conv[0], conv[1],kernel_size=3, stride = 1, padding=1, bias=True)
        self.pool2 = nn.MaxPool1d(4)    
        self.fc1 = nn.Linear(conv[1], 6)
        if self.quantum:
            self.qlayer = qml.qnn.TorchLayer(qnode = self.circuit, weight_shapes = weight_shapes)
        self.fc2 = nn.Linear(6, 1)
        self.fc3 = nn.Linear(1, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        
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
        if self.quantum: 
            x = self.qlayer(x)
            logger.debug("Shape: {}".format(x.shape))
        x = self.fc2(x)
        logger.debug("Shape: {}".format(x.shape))
        output = self.fc3(x)
        logger.debug("Shape: {}".format(output.shape))
        return output

    def train(self) -> None:
        logger.info("Begin training.")
        start = time.time() # Start Timer
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
        self.train_time = time.time() - start # Finish timing 
        logger.debug("Loss: {}".format(loss))

    def test(self) -> None:
        logger.info("Begin testing.")
        start = time.time() # Start Timer
        self.y_pred = self.forward(self.X_test)
        self.y_data_predict = self.y_pred.data.numpy() 
        self.y_data_test = self.y_test.data.numpy()
        self.test_loss = self.criterion(self.y_pred, self.y_test).detach().numpy()
        logger.info("Test Loss: {}".format(self.test_loss))
        self.test_time = time.time() - start # Finish timing 

        logger.debug("Test {}\nPredict{}".format(self.y_data_test, self.y_data_predict))
        logger.debug("Test {}\nPredict{}".format(self.y_data_test.shape, self.y_data_predict.shape))


