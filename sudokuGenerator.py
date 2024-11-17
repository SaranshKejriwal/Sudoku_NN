
# randomize rows, columns and numbers (of valid base pattern)
from random import sample
import numpy as np
import pandas as pd 

'''
I have used the Sudoku generator code from https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python as my reference
'''

base  = 3
side  = base*base








class sudokuGenerator:
    
    # pattern for a baseline valid solution
    def pattern(self, r,c): 
        return (base*(r%base)+r//base+c)%side

    def shuffle(self, s): 
        return sample(s,len(s)) 

    def createSudokuDatasetCSV(self, numTrainingExamples):

        xInputTrainingDataset = np.zeros((numTrainingExamples,81)) #if we want each row to be a separate example, then there need to be 81 columns
        yOutputTrainingDataset = np.zeros((numTrainingExamples,81))

        #create new dataset for validation, and append to the validation csv.
        numValidationExamples = int(numTrainingExamples/10)

        xInputValidationDataset = np.zeros((numValidationExamples,81)) #if we want each row to be a separate example, then there need to be 81 columns
        yOutputValidationDataset = np.zeros((numValidationExamples,81))

        #save training examples
        for i in range(numTrainingExamples):
            yOutputTrainingDataset[i] = self.createSingleSudokuCompletedPuzzle()
            xInputTrainingDataset[i] = self.createPuzzleFromSolution(yOutputTrainingDataset[i])

        #save validation examples
        for i in range(numValidationExamples):
            yOutputValidationDataset[i] = self.createSingleSudokuCompletedPuzzle()
            xInputValidationDataset[i] = self.createPuzzleFromSolution(yOutputValidationDataset[i])
        self.writeSudokuToCSV(xInputTrainingDataset, yOutputTrainingDataset, xInputValidationDataset, yOutputValidationDataset)

        return

    def createSingleSudokuCompletedPuzzle(self):
        rBase = range(base) 
        rows  = [ g*base + r for g in self.shuffle(rBase) for r in self.shuffle(rBase) ] 
        cols  = [ g*base + c for g in self.shuffle(rBase) for c in self.shuffle(rBase) ]
        nums  = self.shuffle(range(1,base*base+1))

        # produce board using randomized baseline pattern
        #board = [ [nums[self.pattern(r,c)] for c in cols] for r in rows ]
        board = np.array([ [nums[self.pattern(r,c)] for c in cols] for r in rows ])
        yOutSingle = board.reshape(81) # the 9x9 board is now valid...create an continuous array from it as the output
        return yOutSingle


    def createPuzzleFromSolution(self, y):

        #this var specifies how much % of the puzzle should be hidden - should ideally be between 2% and 70%
        gridPercentToHide = 0.1 
        #Note - training and test data has examples where we took this value as 0.02 also,meaning that only 1-2 cells are hidden and rest are solved. This is easiest thing a model can do.
        #Harder examples were appended to the existing training and test sets

        #any % the 81 numbers will be set to 0, making this the partially incomplete sudoku
        hiddenCells = np.random.choice(np.arange(y.size), replace=False, size=int(y.size * gridPercentToHide))

        x = np.copy(y) #copy y to x; Note x=y assignment operator creates a direct reference and updates both values.
        x[hiddenCells] = 0 #suppress x at the indices in hiddenCells
        return x

    def writeSudokuToCSV(self, xTrain, yTrain, xVal, yVal):

        inputTrainingFilename = 'datasets/training_input.csv'
        outputTrainingFilename = 'datasets/training_output.csv'

        inputValidationFilename = 'datasets/validation_input.csv'
        outputValidationFilename = 'datasets/validation_output.csv'

        xTrainDF = pd.DataFrame(xTrain)
        yTrainDF = pd.DataFrame(yTrain)

        xValDF = pd.DataFrame(xVal)
        yValDF = pd.DataFrame(yVal)

        #we don't want to overwrite the old training examples, we want to append them to the existing training set, hence mode = 'a'
        xTrainDF.to_csv(inputTrainingFilename, mode='a', index=False, header = False) #header has to be false to ensure that index numbers (0-80) are not appended in the middle of the data.
        yTrainDF.to_csv(outputTrainingFilename, mode='a', index=False, header = False)

        xValDF.to_csv(inputValidationFilename, mode='a', index=False, header = False)
        yValDF.to_csv(outputValidationFilename, mode='a', index=False, header = False)

        return
