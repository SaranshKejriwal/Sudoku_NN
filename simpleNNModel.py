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

    sudokuGridSize = 9 #this will be constant.

    #this is a 9x9 array of networks
    #sudokuSimpleNN = simpleNN[sudokuGridSize][sudokuGridSize]
    sudokuSimpleNN = [[simpleNN() for j in range(9)] for i in range(9)] #2D array

    def trainModel(self, x_train, y_train):
        print("training Simple Neural Net Model with 81 neural nets for each cell...")

        print(y_train)
        print (y_train[:,0])

        for t in range(self.iterations):

            for i in range (self.sudokuGridSize):
                for j in range (self.sudokuGridSize):

                    '''From an array of [m,81], where m is the training example, we need to get the output for each cell individually, since one NN corresponds to one cell. 

                        if you distribute a row of 81 entries on a 9x9 grid, cell (i,j) will contain entry number 9i+j '''
                    y_train_cell_ij = y_train[:,9*i + j] 
                    

                    self.sudokuSimpleNN[i][j].trainModel(x_train, y_train_cell_ij) #each network will receive the entire grid of input sudoku but will focus only on its own cell.

        return 


    #this method initializes the starting weights and biases of the network before training.
    '''def initParams(self):

        W1 = np.random.randn(self.input_layer_neurons,81) #12 neurons of size 81, to align with input matrix
        b1 = np.random.randn(self.input_layer_neurons, ) #Note - adding (x,y) creates a list of lists.

        W2 = np.random.randn(self.hidden_layer_neurons,self.input_layer_neurons) 
        b2 = np.random.randn(self.hidden_layer_neurons, )

        return W1, b1, W2, b2'''

