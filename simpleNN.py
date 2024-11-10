'''
This class contains a single 10 layer neural network.
The simpleNNModel will create 81 of these simpleNN objects, one for each cell in Sudoku.
'''
from math import log
import mathFunctions
import numpy as np

class simpleNN:

    #this will inherit its hyper parameters from the model class. This is only a single neural network
    
    #Note - 12 aand neurons has been selected somewhat randomly for now, just to kick things off....
    input_layer_neurons = 12
    hidden_layer_neurons = 9 # since output will have 9 possible values only.
    
    #this is not a hyperParam. THis value will track the loss within this specific network
    currentLoss = 9999 #initialized as a super high value to track any neurons that remain untrained in case their cell was largely populated.

    

    def __init__(self):
        #constructor

        #initialize the weights and biases of this network with random values from -0.5 to 0.5
        self.W1, self.b1, self.W2, self.b2 = self.initParams()
        return


    def trainModel(self, x_train, y_train):
        #print("training Simple Neural Net Model...")
        numSamples = y_train.size

        Z1, A1, Z2, A2 = self.forwardProp(self.W1, self.b1, self.W2, self.b2, x_train)
        expectedOutputProbability = self.updateLossValue(A2,y_train)

        dW2, dB2, dW1, dB1 = self.backProp(numSamples, expectedOutputProbability, A1,Z1,self.W2,x_train)

        self.updateParams(dW2, dB2, dW1, dB1)

        return 

    def forwardProp(self, W1, b1, W2, b2, x_train):
        #this method takes weights and biases as external args, not from the class properties itself.
        print(W1.shape)
        print (x_train.T.shape)
        print (b1.shape) #12,1
        Z1 = np.add(W1.dot(x_train.T) , b1) #equivalent to Z1 = W1.X1_T + b1

        print(Z1.shape) #12,m

        A1 = mathFunctions.Tanh(Z1)
        print(A1.shape) #12, m
        
        Z2 = np.add(W2.dot(A1) , b2) #equivalent to Z2 = W2.A1 + b1     Note that A1 is NOT transposed.

        print(Z2.shape)

        A2 = mathFunctions.softmax(Z2)
        print(A2.shape)

        return Z1, Z1, Z2, A2

    def updateLossValue(self, A2, y_train):

        #Note - Each neural network applies to an individual cell of the sudoku, so Loss will actually be an 81-dimension array

        yOneHot = mathFunctions.getOneHotVector(y_train)

        #get the indices from the predictions, corresponding to the outputs y of interest.
        print(y_train)
        print(A2)
        print(yOneHot)
        #probabilityOfExpectedOutput = A2[range(yOneHot.size),yOneHot]
        probabilityOfExpectedOutput = np.multiply(A2,yOneHot) #get the probability of ONLY the expected output via element-wise multiplication

        print(probabilityOfExpectedOutput) #test only

        #using negative log likelihood method to calculate loss value for all the training examples
        self.currentLoss = (-1 * sum(log(probabilityOfExpectedOutput)))/probabilityOfExpectedOutput.size

        return probabilityOfExpectedOutput

    def backProp(self, numSamples, A2y, A1, Z1, W2,x_train):

        #Refer to implementation notes in Word document. probabilityOfExpectedOutput is equivalent to A2y

        dW2 = (-1)*(A2y.dot(A1))/numSamples #check implementation notes
        dA1 = (-1)*(A2y.dot(W2))/numSamples #this is an intermediate step used to calculate dW1 and dB1

        dB2 = (-1)*(A2y)/numSamples


        dB1 = dA1.dot(mathFunctions.dTanh(Z1)) #equivalent to d(loss)/d(A1) . d(A1)/d(Z1) 
        #d(A1)/d(Z1) = g'(Z1) = dTanh(Z1)
        #note that dB1 = dZ1 because Z1 = W1.X + B1, so we need not save dZ1 as a separate variable.

        dW1 = dB1.dot(x_train)

        return dW2, dB2, dW1, dB1


    def updateParams(self,dW2, dB2, dW1, dB1):

        self.W2 = self.W2 - dW2
        self.b2 = self.b2 - dB2
        self.W1 = self.W1 - dW1
        self.b1 = self.b1 - dB1

        return


    #this method initializes the starting weights and biases of the network before training.
    def initParams(self):

        W1 = np.random.randn(self.input_layer_neurons,81) #12 neurons of 81 dimensions, to align with input matrix
        b1 = np.random.randn(self.input_layer_neurons, 1) #Note - adding (x,y) creates a list of lists.

        W2 = np.random.randn(self.hidden_layer_neurons,self.input_layer_neurons) 
        b2 = np.random.randn(self.hidden_layer_neurons, 1)

        return W1, b1, W2, b2

