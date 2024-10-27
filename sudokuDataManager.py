'''
This class does the following:
creates Sudoku solved-grids, 
generates puzzles from solved grids,
writes them to static csv files
and/or loads them from pre-written csv files

'''

class sudokuDataManager():


    #this is preliminary sample data generator, that gives one hard coded example for training
    def getSingleInputExample(self):

        x_single_input = [0,4,6,0,1,0,8,5,7,0,9,1,0,0,8,2,0,4,2,8,0,5,0,4,1,3,9,1,3,5,2,0,7,0,8,0,7,0,0,6,8,0,4,1,0,4,6,8,1,0,3,0,7,2,0,1,0,3,0,6,0,4,0,6,0,0,8,2,0,3,0,5,0,0,3,0,0,9,6,0,0]

        return x_single_input


    #this is preliminary sample data generator, that gives one hard coded example for training
    def getSingleOutputExample(self):

        y_single_output = [3,4,6,9,1,2,8,5,7,5,9,1,7,3,8,2,6,4,2,8,7,5,6,4,1,3,9,1,3,5,2,4,7,9,8,6,7,2,9,6,8,5,4,1,3,4,6,8,1,9,3,5,7,2,9,1,2,3,5,6,7,4,8,6,7,4,8,2,1,3,9,5,8,5,3,4,7,9,6,2,1]

        return y_single_output

