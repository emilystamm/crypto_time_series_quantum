"""
===================================================
CONV_MODEL.PY
CryptoTimeSeries base model; inherits from nn.Module
===================================================
"""
# Imports 
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
logger.setLevel(logging.INFO)

from model import CryptoTimeSeriesModel, num_qubits
from utils import layer

# Devices
dev = qml.device("default.qubit", wires=num_qubits)
# s3 = ("amazon-braket-amazon-braket-emily-protiviti", "qhack-rigettim1-conv-2")
# device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1"
# device_arn="arn:aws:braket:::device/qpu/ionq/ionQdevice"
# device_arn = "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-1"
# dev = qml.device("braket.aws.qubit",device_arn = device_arn, s3_destination_folder=s3, wires=num_qubits, shots=100, parallel=True)

"""
ConvCryptoTimeSeriesModel
Convolutional and hybrid class; inherits from CryptoTimeSeriesModel.
"""
class ConvCryptoTimeSeriesModel(CryptoTimeSeriesModel):
    '''
    Init
    '''
    def __init__(self,
        num_train, num_test, iterations, lr, batch_size,start_index, lookback, conv, quantum
    ) -> None:
        super(ConvCryptoTimeSeriesModel, self).__init__(num_train, num_test, iterations, lr, batch_size,start_index, lookback, conv=conv, quantum=quantum)
        if self.quantum: self.type =  "CNN_Quantum"
        else: self.type = "CNN"
    '''
    Initialize Layers 
    '''
    def initialize_layers(self, ):
        logger.info("Initializing layers.")        
        # Set quantum and convolutional parameters
        self.input_channels, self.output_size = 6*(self.lookback- 1), 1 # Conv: Set input / output size 
        self.num_layers = 5 # Q: Set number of layers
        weight_shapes = {"weights": (self.num_layers, num_qubits, 3)} # Q: parameters for entangle layers
        # Create layers for neural network 
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=self.conv[0], kernel_size=6, stride=6, padding=5, bias=True
        )
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(
            in_channels=self.conv[0], out_channels=self.conv[1], kernel_size=3, stride=3, padding=5, bias=True
        )
        self.pool2 = nn.MaxPool1d(3)
        self.fc1 = nn.Linear(self.conv[1], 6)
        # Quantum layer
        if self.quantum: 
            self.qlayer = qml.qnn.TorchLayer(qnode = self.circuit, weight_shapes = weight_shapes)
            # self.qlayer2 = qml.qnn.TorchLayer(qnode = self.circuit2, weight_shapes = weight_shapes)
        # Else use linear layers instead 
        else:
            self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 1)
        # Use Adam optimizer 
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

    '''
    Forward 
    feed x through neural network 
    '''
    def forward(self,x):
        logger.debug("Input Shape: {}".format(x.shape)) # (batch_size, 1, 6*(self.lookback- 1))
        self.input_size = x.size(0)  
        x = self.relu(self.conv1(x))
        logger.debug("Shape: {}".format(x.shape))
        x = self.pool1(x)
        logger.debug("Shape: {}".format(x.shape))
        x = self.relu(self.conv2(x))
        logger.debug("Shape: {}".format(x.shape))
        x = self.pool2(x)
        logger.debug("Shape: {}".format(x.shape))
        x = x.view(self.input_size, -1)  # flatten the tensor
        logger.debug("Shape: {}".format(x.shape))
        x = self.fc1(x)
        logger.debug("Shape: {}".format(x.shape))
        if self.quantum: 
            x = self.qlayer(x)
            logger.debug("Shape: {}".format(x.shape))
            # x = self.qlayer2(x)
            # logger.debug("Shape: {}".format(x.shape))
        else: 
            x = self.fc2(x)
        x = self.fc3(x)
        logger.debug("Shape: {}".format(x.shape))
        return x

    '''
    Train
    '''
    def train(self) -> None:
        logger.info("Begin training.")
        start = time.time() 
        for iteration in range(self.iterations):
            batch_index = np.random.randint(0, self.y_train.shape[0], (self.batch_size))
            X_batch = self.X_train[batch_index]
            y_batch = self.y_train[batch_index]
            outputs = self.forward(X_batch)
            self.optimizer.zero_grad()
            logger.debug("X_batch.shape {} y_batch.shape {}".format(X_batch.shape, y_batch.shape))
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            if iteration % 100 == 0: logger.info("Iteration: {} loss: {}".format(iteration, loss.detach().numpy()))
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
        logger.debug("Call circuit {} {}".format(inputs, weights))
        AngleEmbedding(inputs, wires=range(num_qubits))
        StronglyEntanglingLayers(weights, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]


    '''
    Circuit2
    for alternate quantum layer
    '''
    @qml.qnode(dev)
    def circuit2(inputs, weights):
        AngleEmbedding(inputs, wires=range(num_qubits))
        for weight in weights:
            layer(weight)
        
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]