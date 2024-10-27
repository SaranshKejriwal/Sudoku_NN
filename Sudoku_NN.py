import numpy as np
import pandas as pd
from modelRunner import modelRunner
from simpleNNModel import simpleNNModel
from sudokuDataManager import sudokuDataManager

#this is the initializer class which triggers the modelRunner


print("Hello World")

m = modelRunner()
s = sudokuDataManager()
nn = simpleNNModel()
x_train = s.getSingleInputExample()
y_train = s.getSingleOutputExample()

m.trainModel(nn, x_train, y_train)