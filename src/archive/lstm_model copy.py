import time 
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

# num_qubits = 6
# dev = qml.device("default.qubit.torch", wires=num_qubits)
dev = qml.device("default.qubit", wires=num_qubits)


class LSTMCryptoTimeSeriesModel(CryptoTimeSeriesModel):
    def __init__(self,
        num_train, num_test, epochs, lr, batch_size,start_index, lookback
    ) -> None:
        super(LSTMCryptoTimeSeriesModel, self).__init__(num_train, num_test, epochs, lr, batch_size,start_index, lookback)        
        self.num_hidden_layers=self.batch_size # change 
        self.sequence_length = lookback - 1
        self.hidden_dim = 128
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(1, self.hidden_dim)
        self.lstm2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, 1)
        self.criterion = nn.MSELoss()

    def init_hidden_state(self):
        hidden = (torch.zeros(self.num_hidden_layers, self.batch_size, self.hidden_dim),
                    torch.zeros(self.num_hidden_layers, self.batch_size, self.hidden_dim),
                     torch.zeros(self.num_hidden_layers, self.batch_size, self.hidden_dim),
                      torch.zeros(self.num_hidden_layers, self.batch_size, self.hidden_dim))
        return hidden


    def forward(self, X, prev_state):
        h_t, c_t, h_t2, c_t2 = prev_state
        for t in X.shape[0]:
            h_t, c_t = self.lstm1(X, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
        return output, (h_t2, c_t2)
        # lstm_out, hidden = self.lstm1(X, hidden)
        # outputs, n_samples = [], y.shape[0]
        # h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        # c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        # h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        # c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)

        # for time_step in range(y.shape[0]):
        #     input_t = X[time_step]
        #     h_t, c_t = self.lstm1(input_t, (h_t,c_t))
        #     h_t2, c_t2 = self.lstm2(h_t, (h_t2. c_t2))
        #     ouptut = self.linear(h_t2)
        #     outputs.append(ouptut)
        # for i in range(future_preds):
        #     output  = outputs[i]
        #     h_t,c_t = self.lstm1(output, (h_t,c_t))
        #     h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        #     output = self.linear(h_t2)
        #     outputs.append(output)
        # ouptuts = torch.cat(outputs, dim = 1)
        # return outputs
            

    def train(self) -> None:
        logger.info("Begin training.")
        start = time.time() 
        model = self
        # model.train()
        self.optimizer = torch.optim.Adam(model.parameters(), self.lr)

        for epoch in range(self.epochs):
            logger.debug("Epoch {}".format(epoch))
            state_h, state_c = model.init_hidden_state()
            batch_index = np.random.randint(0, self.y_train.shape[0], (self.batch_size))
            X_batch = self.X_train[batch_index]
            y_batch = self.y_train[batch_index]
            self.optimizer.zero_grad()

            X_batch = torch.reshape(X_batch, (X_batch.shape[0], X_batch.shape[2], X_batch.shape[1]))

            logger.debug("X_batch, (state_h, state_c) \n{} \n{}\n".format(X_batch.shape, (state_h.shape, state_c.shape)))
            y_pred, (state_h, state_c) = self.forward(X_batch, (state_h, state_c))
            loss = self.criterion(y_pred.transpose(1, 2), y_batch)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            self.optimizer.step()

        self.train_time = time.time() - start 

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


    @qml.qnode(dev)
    def circuit(inputs, weights):
        AngleEmbedding(inputs, wires=range(num_qubits))
        StronglyEntanglingLayers(weights, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]