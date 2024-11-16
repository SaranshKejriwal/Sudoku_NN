
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

    def createSudokuDatasetCSV(self, numExamples):

        xInputDataset = np.zeros((numExamples,81)) #if we want each row to be a separate example, then there need to be 81 columns
        yOutputDataset = np.zeros((numExamples,81))

        for i in range(numExamples):
            yOutputDataset[i] = self.createSingleSudokuCompletedPuzzle()
            xInputDataset[i] = self.createPuzzleFromSolution(yOutputDataset[i])

        print(yOutputDataset)

        self.writeSudokuToCSV(xInputDataset, yOutputDataset, numExamples)

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
        print(yOutSingle)
        return yOutSingle


    def createPuzzleFromSolution(self, y):

        #any 56 out of the 81 numbers will be set to 0, making this the partially incomplete sudoku
        hiddenCells = np.random.choice(np.arange(y.size), replace=False, size=int(y.size * 0.7))

        x = np.copy(y) #copy y to x; Note x=y assignment operator creates a direct reference and updates both values.
        x[hiddenCells] = 0 #suppress x at the indices in hiddenCells
        return x

    def writeSudokuToCSV(self, x, y, numExamples):

        inputFilename = 'datasets/input_'+str(numExamples)+'.csv' #create new filenames based on dataset sizes
        outputFilename = 'datasets/output_'+str(numExamples)+'.csv'

        xDF = pd.DataFrame(x)
        yDF = pd.DataFrame(y)

        xDF.to_csv(inputFilename, index=False)
        yDF.to_csv(outputFilename, index=False)

        return
