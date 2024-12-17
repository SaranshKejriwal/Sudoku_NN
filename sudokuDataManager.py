'''
This class does the following:
creates Sudoku solved-grids, 
generates puzzles from solved grids,
writes them to static csv files
and/or loads them from pre-written csv files

'''
import numpy as np
import pandas as pd

class sudokuDataManager():


    #this is preliminary sample data generator, that gives 1 or 2 hard coded example for sanity testing
    def getTwoTrainingInputExamples(self):
        
        x_single_input = np.array([[0,4,6,0,1,0,8,5,7,0,9,1,0,0,8,2,0,4,2,8,0,5,0,4,1,3,9,1,3,5,2,0,7,0,8,0,7,0,0,6,8,0,4,1,0,4,6,8,1,0,3,0,7,2,0,1,0,3,0,6,0,4,0,6,0,0,8,2,0,3,0,5,0,0,3,0,0,9,6,0,0],[4,0,5,0,1,0,3,0,9,0,1,8,0,0,0,7,0,5,0,7,3,9,0,6,8,4,1,8,0,6,3,7,1,2,0,0,3,2,0,0,6,9,0,7,0,1,5,0,2,4,0,6,9,3,6,0,0,7,0,4,5,0,2,0,3,1,6,9,2,4,8,7,0,0,0,1,0,5,9,0,6]])
        return x_single_input


    #this is preliminary sample data generator, that gives 1 or 2 hard coded example for training
    def getTwoTrainingOutputExamples(self):

        y_single_output = np.array([[3,4,6,9,1,2,8,5,7,5,9,1,7,3,8,2,6,4,2,8,7,5,6,4,1,3,9,1,3,5,2,4,7,9,8,6,7,2,9,6,8,5,4,1,3,4,6,8,1,9,3,5,7,2,9,1,2,3,5,6,7,4,8,6,7,4,8,2,1,3,9,5,8,5,3,4,7,9,6,2,1],[4,6,5,8,1,7,3,2,9,9,1,8,4,2,3,7,6,5,2,7,3,9,5,6,8,4,1,8,9,6,3,7,1,2,5,4,3,2,4,5,6,9,1,7,8,1,5,7,2,4,8,6,9,3,6,8,9,7,3,4,5,1,2,5,3,1,6,9,2,4,8,7,7,4,2,1,8,5,9,3,6]])
        return y_single_output

        #this is preliminary sample data generator, that gives 1 or 2 hard coded example for sanity testing
    def getTwoSmallTrainingInputExamples(self):
        
        x_single_input = np.array([[0,3,7,4,6,2,1,8,9],[9,7,3,5,6,0,8,1,4],[1,3,2,7,4,6,9,5,0]])
        return x_single_input


    #this is preliminary sample data generator, that gives 1 or 2 hard coded example for training
    def getTwoSmallTrainingOutputExamples(self):

        y_single_output = np.array([[5,3,7,4,6,2,1,8,9],[9,7,3,5,6,2,8,1,4],[1,3,2,7,4,6,9,5,8]])
        return y_single_output

    def getCSVTrainOutputExamples(self):

        outputData = pd.read_csv('datasets/training_output.csv')
        #Note - numpy reads pandas dataframe as floats by default. These need to be changed to int
        outputData = outputData.astype(int)
        outputData = outputData.astype(int)

        outputData = np.array(outputData) #store numpy array in same variable

        return outputData

    def getCSVTrainInputExamples(self):

        inputData = pd.read_csv('datasets/training_input.csv')

        #Note - numpy reads pandas dataframe as floats by default. These need to be changed to int
        inputData = inputData.astype(int)
        inputData = inputData.astype(int)
        
        inputData = np.array(inputData) #store numpy array in same variable

        return inputData

    def getCSVValidationOutputExamples(self):

        outputData = pd.read_csv('datasets/validation_output.csv')
        #Note - numpy reads pandas dataframe as floats by default. These need to be changed to int
        outputData = outputData.astype(int)
        outputData = outputData.astype(int)

        outputData = np.array(outputData) #store numpy array in same variable

        return outputData

    def getCSVValidationInputExamples(self):

        inputData = pd.read_csv('datasets/validation_input.csv')

        #Note - numpy reads pandas dataframe as floats by default. These need to be changed to int
        inputData = inputData.astype(int)
        inputData = inputData.astype(int)
        
        inputData = np.array(inputData) #store numpy array in same variable

        return inputData
