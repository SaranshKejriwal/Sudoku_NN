'''
This class contains a single 10 layer neural network.
The simpleNNModel will create 81 of these simpleNN objects, one for each cell in Sudoku.
'''
from ast import Num
from colorsys import yiq_to_rgb
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


    def trainModel(self, x_train, y_train, cellPosition):
        
        numSamples = y_train.size

        Z1, A1, Z2, A2 = self.forwardProp(x_train)
        expectedOutputProbability = self.updateLossValue(A2,y_train)

        #print("loss at cell position ", cellPosition)
        #print(self.currentLoss)


        dW2, dB2, dW1, dB1 = self.backProp(numSamples, expectedOutputProbability, A1,Z1,self.W2,x_train)

        self.updateParams(dW2, dB2, dW1, dB1)

        return 

    def forwardProp(self, x_train):
        #this method takes weights and biases as external args, not from the class properties itself.
        
        Z1 = np.add(self.W1.dot(x_train.T) , self.b1) #equivalent to Z1 = W1.X1_T + b1; should be of shape (12,m)

        A1 = mathFunctions.Tanh(Z1)  #12,m
        
        Z2 = np.add(self.W2.dot(A1) , self.b2) #equivalent to Z2 = W2.A1 + b1     Note that A1 is NOT transposed.
        #should be of shape (9,m)
        A2 = mathFunctions.softmax(Z2) #should be of shape (9,m) 

        return Z1, A1, Z2, A2

    def updateLossValue(self, A2, y_train):

        #Note - Each neural network applies to an individual cell of the sudoku, so Loss will actually be an 81-dimension array

        numExamples = np.shape(y_train)[0] #get number of rows in y_train
        yOneHot = mathFunctions.getOneHotVector(y_train)

        #get the indices from the predictions, corresponding to the outputs y of interest.
        probabilityOfExpectedOutput = np.multiply(A2,yOneHot.T) #get the probability of ONLY the expected output via element-wise multiplication


        #using negative log likelihood method to calculate loss value for all the training examples
        try:
            lossVector = -1 * np.log(A2)
            self.currentLoss = np.multiply(lossVector,yOneHot.T).sum()/numExamples #get the loss against the indices of the expected output value.        
        except:
            print("Negative Log Likelihood failed for: ", A2) #needed in case there are any "Not-a-number" issues.
            print("loss Computation failed for: ", lossVector)

        #self.currentLoss = (-1 * np.sum(np.log(probabilityOfExpectedOutput)))
        #Note - Do NOT attempt an element wise multiplication and take a log of that, because most elements there will be 0, and log(0) is -infinity
        

        return probabilityOfExpectedOutput

    def backProp(self, numSamples, A2y, A1, Z1, W2,x_train):

        #Refer to implementation notes in Word document. probabilityOfExpectedOutput is equivalent to A2y

        '''print("A2y ",A2y.shape)
        print("A1 ",A1.shape)
        print("Z1 ",Z1.shape)
        print("W2 ",W2.shape)
        print("x_tr ",x_train.shape)'''

        dW2 = (-1)*(A2y.dot(A1.T))/numSamples #check implementation notes -> A2y is (9,m) and A1 is (12,m); dW2 should be (9,12), same as W2
        dA1 = (-1)*((W2.T).dot(A2y))/numSamples #this is an intermediate step used to calculate dW1 and dB1; dA1 should be (12,2), same as A1

        dB2 = (-1)*(np.sum(A2y, axis=1, keepdims=True))/numSamples # dB2 should be (9,1), same as B2

        dZ1 = np.multiply(dA1,mathFunctions.dTanh(Z1)) #note that this is an element-wise multiplication, not a dot-product
        #size of dZ1 should be (12,m), same as Z1

        dB1 = np.sum(dA1.dot(mathFunctions.dTanh(Z1.T)), axis = 1, keepdims=True) #equivalent to d(loss)/d(A1) . d(A1)/d(Z1) 
        #d(A1)/d(Z1) = g'(Z1) = dTanh(Z1)
        #note that dB1 = dZ1.sum because Z1 = W1.X + B1; Z1 has distinct columns for each training example but b1 doesn't

        #IMPORTANT - keepdims = True ensures that dB1 is of shape (12,1) and not (12,)

        dW1 = dZ1.dot(x_train)

        return dW2, dB2, dW1, dB1


    def updateParams(self,dW2, dB2, dW1, dB1):

        self.W2 = self.W2 - dW2 #should be of shape (9,12)
        self.b2 = self.b2 - dB2 #should be of shape (9,1)
        self.W1 = self.W1 - dW1 #should be of shape (12,81)
        self.b1 = self.b1 - dB1 #should be of shape (12,1)

        return


    #this method initializes the starting weights and biases of the network before training.
    def initParams(self):

        W1 = np.random.randn(self.input_layer_neurons,81) #12 neurons of 81 dimensions, to align with input matrix
        b1 = np.random.randn(self.input_layer_neurons,1) #Note - adding (x,y) creates a list of lists.

        W2 = np.random.randn(self.hidden_layer_neurons,self.input_layer_neurons) 
        b2 = np.random.randn(self.hidden_layer_neurons,1)

        return W1, b1, W2, b2

