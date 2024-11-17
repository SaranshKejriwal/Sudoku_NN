'''this is the backbone class which does the following steps:


'''

from sudokuDataManager import sudokuDataManager
import abstractModel

class modelRunner():

    # trains a fresh model using new training dataset
    def trainModel(self, model, x_train, y_train, numIterations):

        model.trainModel(x_train, y_train, numIterations)

        return

        # trains a fresh model using new training dataset
    def testModel(self, model, x_test, y_test):

        model.testModel(x_test, y_test)

        return

    # imports weights of a pre-trained model from a static file
    def importModelParameters(self, model):


        return

    # provides the output of a model for a given input and validates the output
    def getModelOutput(self, model, x_test, y_test):

        return

    def runModel(self, model):
        
        return