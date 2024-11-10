'''
this script contains all common mathematical functions that will be used across all the models
'''
import numpy as np

def Tanh(z):
    return np.tanh(z)

def dTanh(tanh):
    #Derivative of tanh() is 1- tanh()^2
    return 1 - np.square(tanh)

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))

#this function returns a list, or list of lists, of random values between -0.5 and 0.5
#xDim is the size of the outer list and yDim is the siz of the inner list.
def getGaussianInit(xDim, yDim):
    return np.random.randn(xDim,yDim)

#this takes a digit from 1-9 as an input and returns an array of zeroes such that only the yth value is 1
def getOneHotVector(y):
    yOneHot = np.zeros(9,dtype=int)
    yOneHot[y-1] = 1 #python index starts at 0, hence y-1

    return yOneHot