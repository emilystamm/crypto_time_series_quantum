import time
from matplotlib.pyplot import xkcd 
import numpy as np
import pennylane as qml

import torch 
import torch.nn as nn
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from model import CryptoTimeSeriesModel, num_qubits

dev = qml.device("default.qubit", wires=num_qubits)

class ConvCryptoTimeSeriesModel(CryptoTimeSeriesModel):
    '''
    Init
    '''
    def __init__(self,
        num_train, num_test, epochs, lr, batch_size,start_index, lookback
    ) -> None:
        super(ConvCryptoTimeSeriesModel, self).__init__(num_train, num_test, epochs, lr, batch_size,start_index, lookback)
    '''
    Initialize Layers 
    '''
    def initialize_layers(self, conv = None, quantum = None):
        logger.info("Initializing layers.")
        # Set type
        if quantum: self.type =  "CNN+Quantum"
        else: self.type = "CNN"
        # Set quantum and convolutional parameters
        self.input_channels, self.output_size = 6*(self.lookback- 1), 1 # Conv: Set input / output size 
        self.num_layers = 5 # Q: Set number of layers
        weight_shapes = {"weights": (self.num_layers, num_qubits, 3)} # Q: parameters for entangle layers
        self.conv = conv # Conv: Set conv channel sizes
        self.quantum = quantum  # If True, use quantum layer 
        # Create layers for neural network 
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=conv[0], kernel_size=6, stride=6, padding=5, bias=True
        )
        self.pool1 = nn.MaxPool1d(self.lookback)
        self.fc1 = nn.Linear(conv[0], 6)
        # Quantum layer
        if self.quantum: 
            self.qlayer = qml.qnn.TorchLayer(qnode = self.circuit, weight_shapes = weight_shapes)
        self.fc2 = nn.Linear(6, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

    '''
    Forward 
    feed x through neural network 
    '''
    def forward(self,x):
        logger.debug("Shape: {}".format(x.shape)) # (batch_size, 6*(self.lookback- 1), 1)
        self.input_size = x.size(0)  
        x = self.relu(self.conv1(x))
        logger.debug("Shape: {}".format(x.shape))
        x = self.pool1(x)
        logger.debug("Shape: {}".format(x.shape))
        x = x.view(self.input_size, -1)  # flatten the tensor
        logger.debug("Shape: {}".format(x.shape))
        x = self.fc1(x)
        logger.debug("Shape: {}".format(x.shape))
        if self.quantum: 
            x = self.qlayer(x)
            logger.debug("Shape: {}".format(x.shape))
        x = self.fc2(x)
        logger.debug("Shape: {}".format(x.shape))
        return x

    '''
    Train
    '''
    def train(self) -> None:
        logger.info("Begin training.")
        start = time.time() 
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
        self.train_time = time.time() - start 
        logger.debug("Loss: {}".format(loss))

    '''
    Test
    '''
    def test(self) -> None:
        logger.info("Begin testing.")
        start = time.time() 
        self.y_pred = self.forward(self.X_test)
        self.y_data_predict = self.y_pred.data.numpy() 
        self.y_data_test = self.y_test.data.numpy()
        self.test_loss = self.criterion(self.y_pred, self.y_test).detach().numpy()
        logger.info("Test Loss: {}".format(self.test_loss))
        self.test_time = time.time() - start 

        logger.debug("Test {}\nPredict{}".format(self.y_data_test, self.y_data_predict))
        logger.debug("Test {}\nPredict{}".format(self.y_data_test.shape, self.y_data_predict.shape))

    '''
    Circuit
    for quantum layer
    '''
    @qml.qnode(dev)
    def circuit(inputs, weights):
        AngleEmbedding(inputs, wires=range(num_qubits))
        StronglyEntanglingLayers(weights, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]