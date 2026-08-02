import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys 

data_train = pd.read_csv('train.csv')
data_train = np.array(data_train)
np.random.shuffle(data_train)

data_test = pd.read_csv('test.csv')
data_test = np.array(data_test)

m,n = data_train.shape
data_dev = data_train[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data_train[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]


def init_params():
    #first layer
    w1 = np.random.randn(15, 784)
    b1 = np.random.randn(15,1)
    #second layer
    w2 = np.random.randn(10,15)
    b2 = np.random.randn(10,1)
    return w1,b1,w2,b2
