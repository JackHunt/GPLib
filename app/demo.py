#!/usr/bin/python3
import sys, os
sys.path.append(os.path.realpath('../build'))
import numpy as np

from pyGP import *

def generateData():
    pass

def runRegression():
    X = np.random.rand(3000, 23)
    Y = np.random.rand(3000)
    X_s = np.random.rand(3000, 23)
    Y_s = np.random.rand(3000)

    regressor = GPRegressor()
    mse = regressor.runRegression(X, Y, X_s, Y_s, {'sigma' : 1.0, 'lambda' : 1.0});
    print("Predictive MSE: %s" % mse)

if __name__ == "__main__":
    generateData()
    runRegression()
