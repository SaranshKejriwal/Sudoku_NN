import numpy as np
import pandas as pd
from modelRunner import modelRunner
from simpleNNModel import simpleNNModel
from sudokuDataManager import sudokuDataManager
from sudokuGenerator import sudokuGenerator

#this is the initializer class which triggers the modelRunner


print("Hello World")


sg = sudokuGenerator()

sg.createSudokuDatasetCSV(100)


'''
m = modelRunner()
s = sudokuDataManager()
nn = simpleNNModel()
x_train = s.getSingleInputExample()
y_train = s.getSingleOutputExample()

#test model on SAME data before training - sanity only
m.testModel(nn,x_train,y_train)

m.trainModel(nn, x_train, y_train)

#test model on SAME data after training - sanity only
m.testModel(nn,x_train,y_train)
'''