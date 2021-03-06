# Cryptocurrency Forecasting with Quantum Machine Learning 
This repository contains several (quantum) machine learning algorithms to predict cryptocurrency prices.

## Summary 
Cryptocurrencies are notoriously volatile and challenging to predict, however, there is a great financial incentive in predicting their value. The purpose of this project is to compare quantum and classical machine learning methods for time series forecasting of cryptocurrency. In particular, this project analyzes the value of Ethereum, between 2015 and 2021 [Kaggle Ethereum Data - Arpit Verma](https://www.kaggle.com/varpit94/ethereum-data). There is a base model (model.py) used for loading and preprocessing data, and writing and plotting results, and the specific machine learning models (conv_model.py, q_seq_model.py) inherit from this model and implement functions for training and testing data. This project includes implementations of CNN, CNN with a quantum layer, and a custom variational quantum algorithm. 

## Usage
First, create a [conda environemnt](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)  for PennyLane. Then install requirements:
```bash
pip3 install -r requirements.txt
``` 
To run, script:
```bash
cd src
python3 main.py
``` 

## Data
The data used is the price of the Ethereum cryptocurrency 2015 - 2021, including the Open, High, Low, Close, Adjusted Close, and Volume for each day. The dataset used is [Kaggle Ethereum Data - Arpit Verma](https://www.kaggle.com/varpit94/ethereum-data).

## Software and Services 
Software and services used include:
* PennyLane: for the Hybrid CNN and Variational Algorithm
* Braket: for running on simulators 
* Torch: for CNN and Hybrid CNN
* Scikit-Learn: for preprocessing data
* Matplotlib: for plotting data
* Pandas and Numpy: for representing data

  
## Resources
* https://arxiv.org/abs/2006.14619
* https://controlandlearning.wordpress.com/2020/07/26/pytorch-basics-1d-convolution/
* https://cnvrg.io/pytorch-lstm/
* https://towardsdatascience.com/pytorch-lstms-for-time-series-data-cd16190929d7
* https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
* https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632