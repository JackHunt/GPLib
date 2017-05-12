#!/usr/bin/python3
import sys, os
sys.path.append(os.path.realpath('../build'))

from pyGP import *

def loadData():
    pass

def runRegression():
    regressor = GPRegressor()

if __name__ == "__main__":
    loadData()
    runRegression()
