'''
This is the simplest Neural network model 
which trains data based on standard set of neurons 
'''

from abstractModel import abstractModel
import numpy as np
from simpleNN import simpleNN

class simpleNNModel(abstractModel):

    #hyper-parameters
    alpha_learning_rate = 0.01

    iterations = 2 #number of iterations for which the model should run

    numNetworks = 81 #number of individual simpleNN objects

    #this will eventually be replaced by a 9x9 array of networks
    singleCellNet = simpleNN()


    def trainModel(self, x_train, y_train):
        print("training Simple Neural Net Model with 81 neural nets for each cell...")

        for i in range(self.iterations):
            self.singleCellNet.trainModel(x_train, y_train)



        return 


    #this method initializes the starting weights and biases of the network before training.
    def initParams(self):

        W1 = np.random.randn(self.input_layer_neurons,81) #12 neurons of size 81, to align with input matrix
        b1 = np.random.randn(self.input_layer_neurons, ) #Note - adding (x,y) creates a list of lists.

        W2 = np.random.randn(self.hidden_layer_neurons,self.input_layer_neurons) 
        b2 = np.random.randn(self.hidden_layer_neurons, )

        return W1, b1, W2, b2

