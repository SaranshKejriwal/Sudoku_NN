'''
this script contains all common mathematical functions that will be used across all the models
'''
import numpy as np

def tanh(z):
    return np.tanh(z)

def dTanh(tanh):
    #Derivative of tanh() is 1- tanh()^2
    return 1 - np.square(tanh)

def softmax(z, softmax_axis = 0):
    #z_sum = np.sum(np.exp(z),axis = softmax_axis, keepdims=True)
    #print("z_sum Shape:", z_sum.shape)
    A2 = np.divide(np.exp(z),(np.sum(np.exp(z),axis = softmax_axis, keepdims=True))) 
    #axis=0 argument is important, to ensure that the sum is only along the preactivations of ONE training example, not against the ENTIRE dataset. 
    return A2

#this function returns a list, or list of lists, of random values between -0.5 and 0.5
#xDim is the size of the outer list and yDim is the siz of the inner list.
def getGaussianInit(xDim, yDim):
    return np.random.randn(xDim,yDim)

#this takes a digit from 1-9 as an input and returns an array of zeroes such that only the yth value is 1
def getOneHotVector(y):
    yOneHot = np.zeros((y.size, 9)) #get m x 9 zeroes which will be arranged into 9 columns, hence an (m,9) vector of zeros

    yOneHot[np.arange(y.size), y-1] = 1 #if y = 5, this means that 4th index from 0-4 should be set to 1
    return yOneHot