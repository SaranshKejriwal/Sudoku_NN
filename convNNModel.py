'''
This is the wavenet Neural network model which splits the incoming puzzle into rows, columns and grids, adn then trains data based on standard set of neurons, that will 
'''

from abstractModel import abstractModel
import numpy as np
from convNN import convNN

class convNNModel(abstractModel):

    #hyper-parameters
    sudokuGridSize = 3 #reducing the "grid" from 9x9 to 3x3

    #this is a 9x9 array of networks
    #sudokuSimpleNN = simpleNN[sudokuGridSize][sudokuGridSize]
    sudokuConvNN = [[convNN() for j in range(9)] for i in range(9)] #2D array

    modelLoss = [[999 for j in range(3)] for i in range(3)] #2D array that holds the loss value of the network

    #these can help break the training loop once the average loss starts increasing
    currentIterationMeanLoss = 999
    previousIterationMeanLoss = 999

    def trainModel(self, x_train, y_train, numIterations):
        print("training Convolutional Neural Net Model with 81 neural nets - 1 for each cell...")

        for t in range(numIterations):

            #check if current loss is lower than previous iteration loss, ie the model is still converging
            if(self.previousIterationMeanLoss >= self.currentIterationMeanLoss):
                #model is still converging. Track the loss of the previous iteration
                self.previousIterationMeanLoss = self.currentIterationMeanLoss
            else:
                #model loss is not dropping anymore. No point in training further.
                #Note - we're not training in batches here, so we may expect a steady decline in loss value
                print("Overall Model loss started increasing at iteration ",t)
                print("Lowest loss achieved:", self.previousIterationMeanLoss)
                print("Stopping training...")

                '''
                Can add code here to write the model objects to csv.
                '''

                break


            if((t+1)%50==0):
                print("Completed Iteration ",t+1)
                #print("Model loss by cell: ", self.modelLoss)
                print("Average model loss: ", self.currentIterationMeanLoss)

            for i in range (self.sudokuGridSize):
                for j in range (self.sudokuGridSize):


                    '''From an array of [m,81], where m is the training example, we need to get the output for each cell individually, since one NN corresponds to one cell. 

                        if you distribute a row of 81 entries on a 9x9 grid, cell (i,j) will contain entry number 9i+j '''
                    y_train_cell_ij = y_train[:,self.sudokuGridSize*i + j] #this corresponds to a vector of output values on a single cell across ALL training examples.
                    
                    self.sudokuConvNN[i][j].trainModel(x_train, y_train_cell_ij,(i,j)) #each network will receive the entire grid of input sudoku but will focus only on its own cell.
                    self.modelLoss[i][j] = self.sudokuConvNN[i][j].currentIterationLoss
            #store the average loss value for tracking when it reaches minima
            self.currentIterationMeanLoss = np.mean(self.modelLoss)

        return 


    def testModel(self, x_test, y_test):
        
        accuracy = 0 #this is defined as the number of expected output cells correctly predicted; cells

        predictions = np.zeros(y_test.shape)
        numSamples = np.shape(y_test)[0]

        #prediction step.
        for i in range(self.sudokuGridSize):
            for j in range(self.sudokuGridSize):

                y_test_cell_ij = y_test[:,self.sudokuGridSize*i + j]
                
                #pass only the one test example of X that you want the solution for. Not the ENTIRE set of x input validation data.
                Z1, A1, Z2, A2, Z3, A3 = self.sudokuConvNN[i][j].forwardProp(x_test, numSamples)

                #predictions of the model will be stored in the output of the dense layer.
                #print("Prediction for cell ",self.sudokuGridSize*i + j+1)
                #print(A3)
                predictions[:,self.sudokuGridSize*i + j] = (np.argmax(A3,1, keepdims=False)+1).astype(int) 
                #argmax() returns the INDEX of the value with the highest probability, which we increase by 1 to get the corresponding sudoku answer

                #important to note that the convNN outputs are of shape (m,9), whereas the simpleNN outputs were of shape (9,m)

        
        #print("prediction:\n",predictions)
        #print("actual:\n",y_test)

        #cell by cell compare
        accuracy = np.count_nonzero(predictions == y_test) #returns a count of cells where values are true (False counts as 0)
        #Note that the model is making a prediction for every cell, even those that are already filled. 


        #note - we're comparing each cell in y to each cell in predictions, instead of columns, since each column consists of several cells across several examples, and if one cell is different, then the whole column is considered non-equal
        print("model accuracy: ", accuracy/y_test.size)
        
        return

