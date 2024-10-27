'''
This is the simplest Neural network model 
which trains data based on standard set of neurons 
'''

from abstractModel import abstractModel
import numpy as np

class simpleNNModel(abstractModel):

    #hyper-parameters
    learning_rate = 0.01

    #Note - 12 aand neurons has been selected somewhat randomly for now, just to kick things off....
    input_layer_neurons = 12
    hidden_layer_neurons = 15
    iterations = 1 #number of iterations for which the model should run

    def trainModel(self, x_train, y_train):
        print("training Simple Neural Net Model...")

        W1, b1, W2, b2 = self.initParams()

        for i in range(self.iterations):
            Z1, A1, Z2, A2 = self.forwardProp(W1, b1, W2, b2, x_train)




        return 

    def forwardProp(self, W1, b1, W2, b2, x_train):
        Z1 = W1.dot(x_train.T) + b1

        print("Z1:" + Z1.shape)

        A1 = self.Tanh(Z1)
        print("A1:" + A1.shape)
        
        Z2 = W2.dot(A1) + b2

        print("Z2:" + Z2.shape)

        A2 = self.softmax(Z2)
        print("A2:" + A2.shape)

        return Z1, Z1, Z2, A2


    #this method initializes the starting weights and biases of the network before training.
    def initParams(self):

        W1 = np.random.randn(self.input_layer_neurons,81) #12 neurons of size 81, to align with input matrix
        b1 = np.random.randn(self.input_layer_neurons, 1)

        W2 = np.random.randn(self.hidden_layer_neurons,81) 
        b2 = np.random.randn(self.hidden_layer_neurons, 1)

        return W1, b1, W2, b2

    def Tanh(self, z):
        return np.tanh(z)

    def dTanh(self, tanh):
        #Derivative of tanh() is 1- tanh()^2
        return 1 - np.square(tanh)

    def softmax(self,z):
        return np.exp(z)/np.sum(np.exp(z))