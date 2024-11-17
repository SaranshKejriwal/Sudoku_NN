import numpy as np
import pandas as pd
from modelRunner import modelRunner
from simpleNNModel import simpleNNModel
from sudokuDataManager import sudokuDataManager
from sudokuGenerator import sudokuGenerator

#this is the initializer class which triggers the modelRunner


print("Hello World")

#sudoku generation code
'''
sg = sudokuGenerator()

sg.createSudokuDatasetCSV(1000)
'''

#'''
m = modelRunner()
s = sudokuDataManager()
nn = simpleNNModel()

#these single datapoints risk overfitting, but these are okay to at least curb the initalization losses.
x_single = s.getTwoTrainingInputExamples()
y_single = s.getTwoTrainingOutputExamples()

x_train = s.getCSVTrainInputExamples()
y_train = s.getCSVTrainOutputExamples()

x_val = s.getCSVValidationInputExamples()
y_val = s.getCSVValidationOutputExamples()

#find accuracy before training
m.testModel(nn,x_train,y_train)
#m.testModel(nn, x_single, y_single) #for quick sanity checks

m.trainModel(nn, x_train, y_train,2000)
#m.trainModel(nn, x_single, y_single,400) #for quick sanity checks

#training loss
print('Final Training Data Accuracy:')
m.testModel(nn,x_train,y_train)
#m.testModel(nn, x_single, y_single) #for quick sanity checks

#testing model
print('Final Validation Data Accuracy:')
m.testModel(nn,x_val,y_val)
#'''