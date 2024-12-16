'''
This is the wavenet Neural network model which splits the incoming puzzle into rows, columns and grids, adn then trains data based on standard set of neurons, that will 
'''

from abstractModel import abstractModel
import numpy as np
from wavenetNN import wavenetNN

class wavenetNNModel(abstractModel):

    #hyper-parameters
    sudokuGridSize = 3 #reducing the "grid" from 9x9 to 3x3

    #this is a 9x9 array of networks
    #sudokuSimpleNN = simpleNN[sudokuGridSize][sudokuGridSize]
    sudokuGridWavenetNN = [[wavenetNN() for j in range(9)] for i in range(9)] #2D array
    sudokuRowWavenetNN = [[wavenetNN() for j in range(9)] for i in range(9)] #2D array
    sudokuColumnWavenetNN = [[wavenetNN() for j in range(9)] for i in range(9)] #2D array

    modelLoss = [[999 for j in range(3)] for i in range(3)] #2D array that holds the loss value of the network

    def trainModel(self, x_train, y_train, numIterations):
        print("training Wavenet Neural Net Model with 729 neural nets - 3 for each cell...")

        for t in range(numIterations):

            if((t+1)%50==0):
                print("Completed Iteration ",t+1)
                #print("Model loss by cell: ", self.modelLoss)
                print("Average model loss: ", np.mean(self.modelLoss))

            for i in range (self.sudokuGridSize):
                for j in range (self.sudokuGridSize):


                    '''From an array of [m,81], where m is the training example, we need to get the output for each cell individually, since one NN corresponds to one cell. 

                        if you distribute a row of 81 entries on a 9x9 grid, cell (i,j) will contain entry number 9i+j '''
                    y_train_cell_ij = y_train[:,self.sudokuGridSize*i + j] #this corresponds to a vector of output values on a single cell across ALL training examples.
                    
                    self.sudokuGridWavenetNN[i][j].trainModel(x_train, y_train_cell_ij,(i,j)) #each network will receive the entire grid of input sudoku but will focus only on its own cell.
                    self.modelLoss[i][j] = self.sudokuGridWavenetNN[i][j].currentLoss


        return 


    def testModel(self, x_test, y_test):
        
        accuracy = 0 #this is defined as the number of expected output cells correctly predicted; cells

        predictions = np.zeros(y_test.shape)

        #prediction step.
        for i in range(self.sudokuGridSize):
            for j in range(self.sudokuGridSize):

                y_test_cell_ij = y_test[:,self.sudokuGridSize*i + j] 

                Z1, A1, Z2, A2 = self.sudokuGridWavenetNN[i][j].forwardProp(x_test)

                predictions[:,self.sudokuGridSize*i + j] = (np.argmax(A2,0)+1).astype(int) #argmax() returns the INDEX of the value with the highest probability, which we increase by 1 to get the corresponding sudoku answer

        #cell by cell compare
        accuracy = np.count_nonzero(predictions == y_test) #returns a count of cells where values are true (False counts as 0)

        #note - we're comparing each cell in y to each cell in predictions, instead of columns, since each column consists of several cells across several examples, and if one cell is different, then the whole column is considered non-equal
        print("model accuracy: ", accuracy/y_test.size)
        
        return

