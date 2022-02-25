"""
===================================================
Q_SEQ_MODEL.PY
Quantum sequential model
===================================================
"""
# Imports
import time 
import numpy as np
import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
import pennylane.optimize as optimize
import torch.nn as nn

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from model import CryptoTimeSeriesModel, num_qubits
from utils import square_loss, layer

# num_qubits = 6
dev = qml.device("default.qubit", wires=num_qubits, shots=1000)
# dev = qml.device("default.qubit.torch", wires=num_qubits, shots=100)
# s3 = ("emily-braket-qhack", "test")
# device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1"
# dev = qml.device("braket.aws.qubit",device_arn = device_arn, s3_destination_folder=s3, wires=num_qubits, shots=1)

'''
QSequenceCryptoTimeSeriesModel
Quantum sequential class; inherits from CryptoTimeSeriesModel.
'''
class QSequenceCryptoTimeSeriesModel(CryptoTimeSeriesModel):
    '''
    Init
    '''
    def __init__(self,
        num_train, num_test, iterations, lr, batch_size,start_index, lookback, num_layers
    ) -> None:
        super(QSequenceCryptoTimeSeriesModel, self).__init__(num_train, num_test, iterations, lr, batch_size,start_index, lookback, num_layers=num_layers)
        # self.optimizer = optimize.AdamOptimizer(stepsize=.02) 
        self.optimizer = optimize.NesterovMomentumOptimizer(stepsize=self.lr, momentum=0.9)
        self.type = "QuantumSequenceCircuit"

    '''
    Circuit
    Quantum circuit that gets called. Takes in inputs and weights, 
    returns array result from measuring first three qubits. 
    '''
    @qml.qnode(dev) 
    def circuit(inputs, weights):
        logger.debug("inputs.shape {} weights.shape {}".format(inputs.shape, weights.shape))
        for i in range(inputs.shape[0]):
            AngleEmbedding(inputs[i], wires=range(num_qubits))
            StronglyEntanglingLayers(weights[i], wires=range(num_qubits))
            # Uncomment to use created layer instead 
            # for j in range(weights[i].shape[0]):
            #     layer(weights[i,j])
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(3)]

    '''
    Cost 
    Returns the square loss (cost) of given weights, bias, and batches of X and y.
    '''
    def cost(self, weights, bias, X, y):
        logger.debug("X {}\ny {}".format(X, y))
        # Results from circuit before processing
        predictions_arr = np.array(
            [np.array(self.circuit(inputs=X[i], weights=weights)) for i in range(X.shape[0])]
        )
        # Processed results = predictions 
        predictions = self.postprocess_output(predictions_arr, bias)
        # Calculate and return square loss
        self.train_loss =  square_loss(np.array(y), np.array(predictions))
        return self.train_loss
    '''
    PostProcess Output
    Takes in result from circuit and given bias, and returns predictions.
    '''
    def postprocess_output(self, predictions_arr, bias):
        predictions = []
        for x in range(predictions_arr.shape[0]):
            x_arr = 0
            for i in range(predictions_arr.shape[1]):
                p = predictions_arr[x,i]
                if p >= 0:
                    x_arr += p * 2**-i
                else:  x_arr += 0
            predictions += [x_arr+bias] 
        return predictions

    '''
    Set Data Sample
    Helper function to set X/y train/test. 
    '''
    def set_data_sample(self):     
        self.X_train = self.X_train_1
        self.X_test = self.X_test_1
        self.y_train = self.y_train_1
        self.y_test = self.y_test_1

    '''
    Train 
    '''
    def train(self) -> None:
        self.set_data_sample()
        logger.info("Begin training.")
        start = time.time() 
        # Weights (lookback - 1, num_layers, num_qubits, 3)
        weights = 0.01 * np.random.randn(self.lookback - 1, self.num_layers, num_qubits, 3)
        # Bias
        bias = np.array(0.0)
        # For iteration over one batch out of dataset 
        for iteration in range(self.iterations):
            # Set batch index
            batch_index = np.random.randint(0, self.y_train.shape[0], (self.batch_size))
            # Batches 
            X_batch = self.X_train[batch_index]
            y_batch = self.y_train[batch_index]
            # Printing depdnent on iteration number
            if iteration % 5 == 0 and iteration != 0: 
                logger.info("Iteration {} : {}".format(iteration, self.train_loss))
            logger.debug("weights: {}".format(weights))
            # Optimizer step 
            weights, bias, _, _ = self.optimizer.step(self.cost, weights, bias, X_batch, y_batch)
        self.train_time = time.time() - start 
        self.weights = weights 
        self.bias = bias
        logger.debug("Loss: {}".format(self.train_loss))

    '''
    Test
    '''
    def test(self) -> None:
        logger.info("Begin testing.")
        start = time.time() 
        # Results from circuit
        predictions_arr = np.array([np.array(self.circuit(inputs=x, weights=self.weights)) for x in self.X_test])
        # Process results to get predictions
        predictions = self.postprocess_output(predictions_arr, bias=self.bias)
        # Set y_pred 
        self.y_pred = np.array(predictions)
        logger.debug(self.y_pred.shape)
        logger.debug(self.X_test.shape)
        # Get numpy array versions
        self.y_data_test = np.array(self.y_test)
        self.y_data_predict = np.array(self.y_pred)
        # Test data square loss
        self.test_loss = round(float(square_loss(self.y_data_test, self.y_data_predict)),5)
        logger.info("Test Loss: {}".format(self.test_loss))
        self.test_time = time.time() - start 


