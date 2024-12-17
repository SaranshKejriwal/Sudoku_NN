import numpy as np
import pandas as pd
from modelRunner import modelRunner
from simpleNNModel import simpleNNModel
from wavenetNNModel import wavenetNNModel
from convNNModel import convNNModel
from sudokuDataManager import sudokuDataManager
from sudokuGenerator import sudokuGenerator

#this is the initializer class which triggers the modelRunner


print("Hello World")

#sudoku generation code
'''
sg = sudokuGenerator()

sg.createMiniBlockDatasetCSV(500)
'''

#'''
m = modelRunner()
s = sudokuDataManager()
#nn = simpleNNModel()
nn = convNNModel()

#these single datapoints risk overfitting, but these are okay to at least curb the initalization losses.
#x_single = s.getTwoTrainingInputExamples()
#y_single = s.getTwoTrainingOutputExamples()
x_single = s.getTwoSmallTrainingInputExamples() #if testing for 3x3
y_single = s.getTwoSmallTrainingOutputExamples()

x_train = s.getCSVTrainInputExamples()
y_train = s.getCSVTrainOutputExamples()

x_val = s.getCSVValidationInputExamples()
y_val = s.getCSVValidationOutputExamples()










def runModelTrainTestOnFullData(modelRunner, model, x_train, y_train, x_val, y_val, numIterations):
    #find accuracy on training data before training
    modelRunner.testModel(model,x_train,y_train)
    #pre-train accuracy on training data
    m.trainModel(nn, x_train, y_train,2000)

    #testing model
    print('Final Validation Data Accuracy:')
    m.testModel(nn,x_val,y_val)

    #training loss
    print('Final Training Data Accuracy:')
    m.testModel(nn,x_train,y_train)

    return

#for quick sanity checks
def performModelSanityOnSmallData(modelRunner, model,x_single, y_single, numIterations):

    #find accuracy before training
    modelRunner.testModel(model, x_single, y_single) 
    #train model
    m.trainModel(nn, x_single, y_single,numIterations) 

    #post-training accuracy
    m.testModel(nn, x_single, y_single) #for quick sanity checks

    return


runModelTrainTestOnFullData(m,nn,x_train, y_train, x_val, y_val, 2000)

#performModelSanityOnSmallData(m,nn,x_single, y_single, 10000)