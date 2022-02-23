# Crypto Forecasting with Quantum Machine Learning
This repository contains several (quantum) machine learning algorithms to predict cryptocurrency prices.

## Summary 
Cryptocurrencies are notoriously volatile and challenging to predict, however, there is a great financial incentive in predicting their value. The purpose of this project is to compare quantum and classical machine learning methods for time series forecasting of cryptocurrency. In particular, this project analyzes the value of Ethereum, between 2015 and 2021 (Kaggle Ethereum Data - Arpit Verma)[https://www.kaggle.com/varpit94/ethereum-data]. There is a base model (model.py) used for loading and preprocessing data, and writing and plotting results, and the specific machine learning models (conv_model.py, q_seq_model.py) inherit from this model and implment functions for training and testing data. Currently, CNN, CNN with a quantum layer, and a self-designed variational quantum model are in progress, all using PennyLane and Torch, and if time permits, QRNN and LSTM will be implemented as well. 

## Resource Estimate 
The models use 6 logicial qubits, and could run on IonQ, Rigetti, or simulators. 

### Data
The data used is the price of the Ethereum cryptocurrency 2015 - 2021, including the Open, High, Low, Close, Adjusted Close, and Volume for each day. The dataset used is (Kaggle Ethereum Data - Arpit Verma)[https://www.kaggle.com/varpit94/ethereum-data].

### Libraries
Libraries used include:
* PennyLane
* Torch
* Scikit-learn
* Matplotlib
* Numpy

  
## Resources
* https://arxiv.org/abs/2006.14619
* https://controlandlearning.wordpress.com/2020/07/26/pytorch-basics-1d-convolution/
* https://cnvrg.io/pytorch-lstm/
* https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632