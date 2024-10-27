'''
This is a simple interface that all model classes must inherit.

'''


class abstractModel():

    #for triggering forward-Propagation, back-Propagation, param Update
    def trainModel(self, x_train, y_train):
        pass

    #initializes model params with random values;
    def initParams(self):
        return
