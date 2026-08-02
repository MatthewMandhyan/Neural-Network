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

def ReLU(Z):
    return np.maxium(0,Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def forward_propogation(w1,b1,w2,b2,X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = softmax(Z2)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arrange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def derivative_ReLU(Z):
    return Z > 0

def back_propogation(Z1,A1,Z2,A2,W2,Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot(Y)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, 2)
    dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1)
    dW2 = 1 / m * dZ2.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, 2)
    return dW1, db1, dW2, db2

def update_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha):
    w1 = w1 - alpha * dW1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dW2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

