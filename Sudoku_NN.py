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
x_train = s.getCSVTrainInputExamples()
y_train = s.getCSVTrainOutputExamples()

x_val = s.getCSVValidationInputExamples()
y_val = s.getCSVValidationOutputExamples()

#test model on SAME data before training - sanity only
m.testModel(nn,x_train,y_train)

m.trainModel(nn, x_train, y_train)

#training loss
print('Final Training Data Accuracy:')
m.testModel(nn,x_train,y_train)

#testing model
print('Final Validation Data Accuracy:')
m.testModel(nn,x_train,y_train)
#'''