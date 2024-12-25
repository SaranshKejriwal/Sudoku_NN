'''
This class contains a single 10 layer neural network.
The wavenetNN will create 729 of these simpleNN objects, 3 for each cell in Sudoku - one each for grid, row and column.

Each of these wavenet networks will be needed to predict the missing number among 9 - either in row, column or grid - which is relatively achievable in a neural network
'''
from math import isnan, log
import mathFunctions
import numpy as np
from convNNClasses import kernelArray

class convNN:

    #This is only a single neural network for a single cell

    #Note - 12 aand neurons has been selected somewhat randomly for now, just to kick things off....
    hidden_layer_neurons = 12 #this layer exists after the convolution completion and activation - "dense" layer
    
    output_layer_neurons = 9 
    #This will always be 9, since output will always have 9 possible values only, regardless.
    
    #this is not a hyperParam. THis value will track the loss within this specific network
    currentIterationLoss = 999 #initialized as a super high value to track any neurons that remain untrained.
    previousIterationLoss = 9999 #used to stop the training once a local minima is reached.

    learningRate = 0.01 #for updating params
    
    x_length = 3 #we're either passing 9x9 sudokus or 3x3 sub grid
    kernelSize = 2 #2x2 for 3x3 sub grid, and 3x3 for 9x9 sudoku
    kernelCount = 20

    isTrained = False #used to print the message once and then never go into train loop again.
    isSlowTraining = False #used to determine when the learning rate is reduced for slow traversal
    #if learning rate is slowed and the previous iteration loss is less than current iteration, then training should stop.

    def __init__(self):
        #constructor

        #create a 2x2 conv filter for a 3x3 sub-sudoku
        self.kernels = kernelArray(self.kernelSize,self.kernelCount)


        #initialize the weights and biases of the standard network AFTER convolution with random values from -0.5 to 0.5
        self.W1, self.b1, self.W2, self.b2 = self.initParams()

        

        '''
        W1 is 12x81
        b1 is 12x1
        W2 is 9x12
        b2 is 9x1
        '''



        return


    def trainModel(self, x_train, y_train, cellPosition):
        
        #'''
        if(self.isTrained):
            return #minimum loss is achieved. No more training required.

        if(not self.isSlowTraining and self.previousIterationLoss < self.currentIterationLoss):
            print("Reducing Learning Rate for:", cellPosition)
            self.reduceLearningRate()
            self.isSlowTraining = True
            #reduce the learning rate once the loss starts increasing
        #stop training if min loss is achieved AFTER slow training
        elif(self.isSlowTraining and self.previousIterationLoss < self.currentIterationLoss):
            print("Lowest Loss achieved for cell:", cellPosition)
            print("Lowest Loss:",self.previousIterationLoss)            
            self.isTrained = True #stop training once lowest loss is achieved
            return 
        #'''


        numSamples = np.shape(y_train)[0] #y.size will give number of cells, not number of rows.

        Z1, A1, Z2, A2, Z3, A3 = self.forwardProp(x_train, numSamples)
        expectedOutputProbability = self.updateLossValue(A3,y_train, numSamples)


        #reduce learning rate if loss is getting lower
        #self.adaptLearningRate(cellPosition)

        if(isnan(self.currentIterationLoss)): # just for seeing any issues in the logs
            print("Non-numeric Loss found at cell level. Printing parameters:")
            print("Cell position: ", cellPosition)
            print("Z1 ",Z1)
            print("A1 ",A1)
            print("Z2 ",Z2)
            print("A2 ",A2)

        dW2, dB2, dW1, dB1, dK1 = self.backProp(numSamples, expectedOutputProbability, A2, Z2, A1,Z1,self.W2, self.W1,x_train)

        self.updateParams(dW2, dB2, dW1, dB1, dK1)

        return 

    def forwardProp(self, x_train, numSamples):

        #convert the x_train into a square matrix, either 9x9 or 3x3, to perform the convolution
        x_train = np.reshape(x_train,(-1,3,3)) #need to change to (-1,9,9) when doing the full sudoku

        #normalize x values to [0,1]
        x_train = x_train/9
        
        #print("x_train:",x_train.shape)
        #print("x_train:",x_train)
        #print("kernels",self.kernels.getKernelArray())
        #Step 1 - Cross Correlation
        Z1 = self.kernels.getInnerCrossCorrelation(x_train, numSamples)
        #print("Z1",Z1)

        #print("Z1 shape after convolution:",Z1.shape)

        #Normalization required at this step for smoothing. Because Z1 values in just 10 iterations will start to touch infinity
        #Z1_max = np.absolute(Z1).max(0,keepdims=True) #for one training example, Z1 is of shape (12,1) -> take the max ABSOLUTE value across all +ive and -ive weights

        #Z1_normalized = np.divide(Z1,Z1_max) #this ensures that pre-activations are <1...ensuring that tanh() and softmax() combination does not cause +infinity values (especially in softmax)

        #print("Z1", Z1)
        #print("Z1_max", Z1_max)
        #print("Z1_norm", Z1_normalized)

        #Activte the Convolution/Cross Correlation output using tanh()
        A1 = mathFunctions.tanh(Z1)  #12,m

        #print("A1 Before Reshape:",A1.shape)
        #print(A1)
        #reshape the output into a column vector for each training example... and pass to a standard neural network
        A1 = np.reshape(A1,(numSamples,-1)) #A1 was (m,x - k + 1, x - k +1) -> reshaped to (m, 2(x - k + 1))
        #print("A1 After Reshape:",A1.shape)
        
        #input to the hidden layer of h neurons   
        Z2 = np.add(A1.dot(self.W1) , self.b1) #equivalent to Z2 = W1.A1 + b1     Note that A1 is NOT transposed.
        #print("Z2:",Z2.shape)

        A2 = mathFunctions.tanh(Z2) #should be of shape (9,m) 

        #pass the dense layer activation to the output layer
        Z3 =  np.add(A2.dot(self.W2) , self.b2)
        #print("Z3:",Z3.shape)
        A3 = mathFunctions.softmax(Z3, softmax_axis=1) 
        #need to ensure that axis of softmax is along the unique examples, not along the dataset row
        # In the simpleNN, the output was of shape (9,m) whereas here it is (m,9), so the axis needs to be changed
        #print("Z3 Sample:", Z3[0])
        #print("A3 Sample:", A3[0])
        return Z1, A1, Z2, A2, Z3, A3

    def updateLossValue(self, A3, y_train, numSamples):

        #start by assigning the loss computed from the previous iteration.
        self.previousIterationLoss = self.currentIterationLoss

        #Note - Each neural network applies to an individual cell of the sudoku, so Loss will actually be an 81-dimension array
        yOneHot = mathFunctions.getOneHotVector(y_train)

        #get the indices from the predictions, corresponding to the outputs y of interest.
        probabilityOfExpectedOutput = np.multiply(A3,yOneHot) #get the probability of ONLY the expected output via element-wise multiplication

        #using negative log likelihood method to calculate loss value for all the training examples
        try:
            lossVector = -1 * np.log10(A3)
            #get the loss against the indices of the expected output value.
            self.currentIterationLoss = np.sum((np.multiply(lossVector,yOneHot)))/numSamples
            #update the current loss which will be compared to previous loss during training.

            #self.currentLoss = (-1 * np.sum(np.log(probabilityOfExpectedOutput)))/numExamples
            #Note - Do NOT attempt an element wise multiplication and THEN take a log of that, because most elements there will be 0, and log(0) is -infinity

        except Exception as e:
            print("Negative Log Likelihood failed for: ", A3.shape) #needed in case there are any "Not-a-number" issues.      
            print(e)

        return probabilityOfExpectedOutput

    def backProp(self, numSamples, A3y, A2, Z2, A1, Z1, W2, W1,x_train):

        #backprop on the dense layer will be similar to that of the simple NN implementation, 
        #except that A1 and Z1 are now the params of the convolution layer, and NOT the hidden layer,
        #and A2 and Z2 are now the params of the hidden layer, not the output layer

        #Note the flip in dimensions between simpleNN and convNN - backprop multiplications will have to change accordingly

        dW2 = (-1)*(A2.T.dot(A3y))/numSamples #check implementation notes -> A3y is (m,9) and A2 is (m,h); dW2 should be (h,9), same as W2
        dA2 = (-1)*((A3y).dot(W2.T))/numSamples #this is an intermediate step used to calculate dW1 and dB1; dA1 should be (h,m), same as A1

        #print("A3y:",A3y.shape)
        dB2 = (-1)*(np.sum(A3y, axis=0, keepdims=True))/numSamples # dB2 should be (9,1), same as B2
        #print("dB2:",dB2.shape)
        dZ2 = np.multiply(dA2,mathFunctions.dTanh(Z2)) #note that this is an element-wise multiplication, not a dot-product
        #size of dZ1 should be (m,h), same as Z1

        #print("dA2:",dA2.shape)
        #print("Z2:",Z2.shape)

        #dB1 should be of size (h,1)
        dB1 = np.sum(dA2.T.dot(mathFunctions.dTanh(Z2)), axis = 0, keepdims=True) 
        
        #print("dB1:",dB1.shape)
        #IMPORTANT - keepdims = True ensures that dB1 is of shape (12,1) and not (12,)

        #equivalent to d(loss)/d(A2) . d(A2)/d(Z2) 
        #d(A2)/d(Z2) = g'(Z2) = dTanh(Z2)
        #note that dB1 = dZ2.sum because Z1 = W1.X + B1; Z1 has distinct columns for each training example but b1 doesn't

                
        #dW1 = d(L)/d(W1) = d(L)/d(Z2) x d(Z2)/d(W1) = dZ2 . A1
        dW1 = A1.T.dot(dZ2) #dW1 should be of size ((x-k-1)(x-k-1),h) same as W1

        #this intermediary step will help in calculating gradient of loss with respect to the kernel
        dA1 = dZ2.dot(W1.T)
        
        #print(A1)
        kernelSize = self.kernels.getKernelSize()
        #reshape the output into a column vector for each training example... and pass to a standard neural network
        dA1 = np.reshape(dA1,(numSamples, -1,kernelSize,kernelSize)) #A1 was (m,x - k + 1, x - k +1) -> reshaped to (m, 2(x - k + 1))
        
        #print("dA1:", dA1.shape)

        #convert the x_train into a square matrix, either 9x9 or 3x3, to perform the convolution
        x_train = np.reshape(x_train,(-1,3,3)) #need to change to (-1,9,9) when doing the full sudoku

        #Next step - Get the derivative of the loss with respect to convolution layer
        dK1 = self.performBackpropInnerCrossCorrelation(x_train, dA1, numSamples) #should of 2x2 or 3x3, same as the kernel size
        
        #print("dK1:", dK1.shape)

        return dW2, dB2, dW1, dB1, dK1

    #this method updates params after backprop
    def updateParams(self,dW2, dB2, dW1, dB1, dK1):

        self.W2 = self.W2 - self.learningRate * dW2 #should be of shape (9,12)
        self.b2 = self.b2 - self.learningRate * dB2 #should be of shape (9,1)
        self.W1 = self.W1 - self.learningRate * dW1 #should be of shape (12,81)
        self.b1 = self.b1 - self.learningRate * dB1 #should be of shape (12,1)
        self.kernels.updateKernelArray(dK1, self.learningRate) #should be of shape (2x2)

        '''
        print("W2",self.W2.shape)
        print("b2",self.b2.shape)
        print("W1",self.W1.shape)
        print("b1",self.b1.shape)
        '''
        return

    #this method reduces the learning rate for each cell's network after its loss is low enough to not require large jumps
    def reduceLearningRate(self):
        self.learningRate = self.learningRate/10 #reduced to one tenth for slower progression
        print("Reducing learning rate to ", self.learningRate)
        return
    #note - we should eventually add code to compare current loss with previous loss also.


        #this method performs convolution for Backprop, 
        #such that it'll add the convolution result for each training example to compute the loss gradient wrt the filters.
    def performBackpropInnerCrossCorrelation(self, x, convGradient, numExamples):

        inputSideLength = np.shape(x)[1] #check if input is 3x3 or 9x9
        filterSideLength = self.kernels.getKernelSize() #check if filter is 2x2 or 3x3

        #print("shape of x during convolution", x.shape)

        #if a small filter square of side s moves over a big input square of side N, 
        #it can move (N - s + 1) times before it will spill over.
        convOutputLength = inputSideLength - filterSideLength + 1

        #Z = np.zeros((np.shape(x)[0], convOutputLength,convOutputLength)) #this is the expected size of the output after convolution -> (m,2,2)
        
        #derivative of Loss wrt kernels should have the same size as the kernels
        dK1 = np.zeros((self.kernels.getKernelCount(),  self.kernels.getKernelSize(),self.kernels.getKernelSize()))
        
        for t in range(numExamples):

            #define the input value for current example.
            xT = x[t]
            #define the convolution gradient for current example
            dAT = convGradient[t]

            #create temporary variable to hold convolution output of that example
            convT = np.zeros((self.kernels.getKernelSize(),self.kernels.getKernelSize()))

            for i in range(convOutputLength):
                for j in range(convOutputLength):

                    #Add the gradient from each of the training examples...
                    convT[i,j] = np.sum(np.multiply(xT[i:i+2,j:j+2],dAT)) #get the i'th and j'th subarray from x of size (filterSideLength,filterSideLength)
            
            #add the filter gradient of that example to dK1
            dK1 += convT
        return dK1


    #this method initializes the starting weights and biases of the network before training.
    def initParams(self):

        #Note - we want to curb the initialization loss by ensuring that the initial random weights don't end up taking extreme values that are "confidently wrong". Initial biases can be set to 0.
        #Weights will multiplied by 0.01 to ensure that the initialization of the weights is as close to 0 as possible, and the network is basically making random guesses without training.

        conv_output_length = (self.x_length - self.kernelSize + 1) #this is the expected length of the side if the convolution output.

        #W1 needs to take input-size according to the output of the convolution step.
        W1 = np.random.randn(conv_output_length*conv_output_length*self.kernelCount, self.hidden_layer_neurons)  * 0.1 
        #note that these neurons are of size ((x-k+1)*(x-k+1),h) 
        #because we will convert the convolution output into a row vector and feed it to the "dense" layer.
        
        #b1 = np.random.randn(self.input_layer_neurons,1) #Note - adding (x,y) creates a list of lists.
        b1 = np.zeros((self.hidden_layer_neurons)) #initialized at zeros to avoid initialization bias.

        W2 = np.random.randn(self.hidden_layer_neurons, self.output_layer_neurons)  * 0.1
        #output_layer will always have 9 neurons - size of this layer is (h,9) to allow dot product A2.W2 + b2
        
        #b2 = np.random.randn(self.hidden_layer_neurons,1)
        b2 = np.zeros((self.output_layer_neurons))

        #Note - Loss will start increasing if Learning rate is too high

        return W1, b1, W2, b2
