'''
This is the simplest Neural network model 
which trains data based on standard set of neurons 
'''

from abstractModel import abstractModel
import numpy as np

class simpleNNModel(abstractModel):

    #hyper-parameters
    learning_rate = 0.01

    def trainModel(self, x_train, y_train):
        print("training Simple Neural Net Model...")



        return 


    def initParams(self):

        W1 = np.random.randn(12,81) #12 neurons of size 81, to align with input matrix
        #Note - 12 neurons has been selected somewhat randomly for now, just to kick things off....



        return super().initParams()