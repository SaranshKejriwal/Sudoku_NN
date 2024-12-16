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
            #initialize kernels with random values between -0.5 and 0.5
            kernelArray = np.random.randn(self.kernelSize,self.kernelSize)

        return kernelArray

    def getKernelArray(self):
        return self.kernels

    def setKernelArray(self,kernelArray):
        self.kernels = kernelArray

    def getKernelSize(self):
        return self.kernelSize

    def getKernelCount(self):
        return self.getKernelCount




class neuronLayer:
    
    layerSize = 0

    def __init__(self, layerSize):
        self.layerSize = layerSize
    

