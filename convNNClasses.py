from ast import Num
from asyncio.windows_events import NULL
from math import isnan, log
import mathFunctions
import numpy as np

#this class maintains the kernel array used in each of the conolutions
class kernelArray:

    kernelCount = 1 #number of kernels in the conv layer
    kernelSize = 3 #size 3 needed for 9x9 sudoku
    kernels = NULL

    def __init__(self, kernelSize = 3, kernelCount = 1):
        self.kernelSize = kernelSize
        self.kernelCount = kernelCount

        self.kernels = self.initializeKernelArray()

    def initializeKernelArray(self):
        #create a 3D array that represents an array of 2D kernels
        kernelArray = np.zeros((self.kernelCount,self.kernelSize,self.kernelSize))

        for i in range(self.kernelCount):
            #initialize each kernels with random values between -0.5 and 0.5
            kernelArray[i] = np.random.randn(self.kernelSize,self.kernelSize)

        return kernelArray

    def getInnerCrossCorrelation(self, xInput, numSamples):
        inputSideLength = np.shape(xInput)[1] #check if input is 3x3 or 9x9

        convOutputLength = inputSideLength - self.kernelSize + 1 #size of the convolution output = x-k+1
        
        Z = np.zeros((np.shape(xInput)[0], self.kernelCount, convOutputLength,convOutputLength)) 
        #this is the expected size of the output after convolution -> (m,k,2,2)
        for n in range(numSamples):
            for k in range(self.kernelCount):
                for i in range(convOutputLength):
                    for j in range(convOutputLength): #to be replaced with numpy conv2d method for performance.
                        Z[n, k,i,j] = np.sum(np.multiply(xInput[n,i:i+self.kernelSize,j:j+self.kernelSize],self.kernels[k,:])) 
                        #get the i'th and j'th subarray from x of size (filterSideLength,filterSideLength)
        return Z



    def getKernelArray(self):
        return self.kernels

    def updateKernelArray(self, kernelGradient, learningRate):
        if(self.kernels.shape != kernelGradient.shape):
            print("Error: Shape mismatch during Bias update")
            return

        self.kernels = self.kernels - kernelGradient*learningRate

    def getKernelSize(self):
        return self.kernelSize

    def getKernelCount(self):
        return self.kernelCount




class neuronLayer:
    
    numNeurons = 0 #number of neurons in the layer
    weightsArray = NULL
    biasArray = NULL

    def __init__(self, layerNeuronCount, expectedInputCount, initNormalizationFactor = 0.1):
        self.numNeurons = layerNeuronCount #represents the number of neurons, ie, the number of outputs expected

        #Create the weight matrix of the neuron layer based on the number of inputs expected and the number of outputs expected
        self.weightsArray = np.random.randn(expectedInputCount,self.numNeurons) * initNormalizationFactor
        
        #Bias matrix will depend on number of neurons only, ie only the number of outputs.
        self.biasArray = np.zeros(self.numNeurons) #initialized at 0. Will be updated during training
        return
    
    def getNeuronLayerOutput(self, inputArray):
        return np.add(inputArray.dot(self.weightsArray), self.biasArray)


    def updateWeights(self, weightGradientArray):
        if(self.weightsArray.shape != weightGradientArray.shape):
            print("Error: Shape mismatch during Weights update")
            return

        self.weightsArray = self.weightsArray - weightGradientArray
        return

    def updateBiases(self, biasGradientArray):
        if(self.biasArray.shape != biasGradientArray.shape):
            print("Error: Shape mismatch during Bias update")
            return

        self.biasArray = self.biasArray - biasGradientArray
        return

    #Getters
    def getWeightsArray(self):
        return self.weightsArray

    def getBiasArray(self):
        return self.biasArray

    

