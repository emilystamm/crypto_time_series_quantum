import time 
import numpy as np
import pennylane as qml
from pennylane import QNode
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
import pennylane.optimize as optimize

import torch 
import torch.nn as nn

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from model import CryptoTimeSeriesModel
from utils import square_loss


num_qubits = 6
# dev = qml.device("default.qubit.torch", wires=num_qubits, shots=1000)
s3 = ("emily-braket-qhack", "test")
device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1"
dev = qml.device("braket.aws.qubit",device_arn = device_arn, s3_destination_folder=s3, wires=num_qubits, shots=1)


class QSequenceCryptoTimeSeriesModel(CryptoTimeSeriesModel):
    def __init__(self,
        num_train, num_test, epochs, lr, batch_size,start_index, lookback
    ) -> None:
        super(QSequenceCryptoTimeSeriesModel, self).__init__(num_train, num_test, epochs, lr, batch_size,start_index, lookback)
        # self.optimizer = optimize.AdamOptimizer(stepsize=.02) 
        self.optimizer = optimize.NesterovMomentumOptimizer(stepsize=0.01, momentum=0.9)
        self.type = "QuantumSequenceCircuit"

    def initialize(self):
        self.X_train = self.X_train_orig
        self.X_test = self.X_test_orig
        self.y_train = self.y_train_orig
        self.y_test = self.y_test_orig
        self.loss = 1000000
        self.num_layers = 2

    @qml.qnode(dev) 
    def circuit(inputs, weights):
        for i in range(inputs.shape[0]):
            AngleEmbedding(inputs[i], wires=range(num_qubits))
            StronglyEntanglingLayers(weights[i], wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]


    def cost(self, weights, X, y):
        logger.debug("X {}\ny {}".format(X, y))
        predictions_arr = np.array([qml.math.to_numpy(self.circuit(inputs=x, weights=weights)) for x in X])
        predictions = []
        # TODO: update this 
        for x in range(predictions_arr.shape[0]):
            x_arr = 0
            for i in range(predictions_arr.shape[1]):
                p = predictions_arr[x,i]
                if p >= 0:
                    x_arr += p * 2**-i
                else:  x_arr += 0
            predictions += [x_arr]
        sqloss =  square_loss(np.array(y), np.array(predictions))
        self.train_loss = sqloss
        return sqloss
    
    def train(self) -> None:
        logger.info("Begin training.")
        start = time.time() 
        weights = 0.01 * np.random.randn(self.lookback - 1, self.num_layers, num_qubits, 3)
        for epoch in range(self.epochs):
            batch_index = np.random.randint(0, self.y_train.shape[0], (self.batch_size))
            X_batch = self.X_train[batch_index]
            y_batch = self.y_train[batch_index]
            if epoch % 10 == 0: 
                logger.info("Epoch {} {}".format(epoch, round(float(self.loss), 5)))

            logger.debug("weights: {}".format(weights))
            logger.debug("optimizer: {}".format(self.optimizer))
            weights, _, _ = self.optimizer.step(self.cost, weights, X_batch, y_batch)

        self.train_time = time.time() - start 
        logger.debug("Loss: {}".format(self.train_loss))
        self.weights = weights 

    def test(self) -> None:
        logger.info("Begin testing.")
        start = time.time() 
        predictions_arr = np.array([qml.math.to_numpy(self.circuit(inputs=x, weights=self.weights)) for x in self.X_test])
        predictions = []
        for x in range(predictions_arr.shape[0]):
            x_arr = 0
            for i in range(predictions_arr.shape[1]):
                p = predictions_arr[x,i]
                if p >= 0:
                    x_arr += p * 2**-i
                else:  x_arr += 0
            predictions += [x_arr]
        self.y_pred = np.array(predictions)
        logger.debug(self.y_pred.shape)
        logger.debug(self.X_test.shape)

        self.y_data_test = np.array(self.y_test)
        self.y_data_predict = np.array(self.y_pred)
        self.test_loss = round(float(square_loss(self.y_data_test, self.y_data_predict)),5)

        logger.info("Test Loss: {}".format(self.test_loss))
        self.test_time = time.time() - start 
