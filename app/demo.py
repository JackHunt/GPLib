#!/usr/bin/python3
import sys, os
sys.path.append(os.path.realpath('../build'))
import numpy as np
import matplotlib.pyplot as pp

from pyGP import *

X = []
Y = []
X_s = []
Y_s = []
means = []
sd = []

def generateData(numTrain, numTest):
    global X, Y, X_s, Y_s
    X = np.random.uniform(-10, 10, size=(numTrain, 1))
    Y = np.sin(X).flatten()
    X_s = np.linspace(-10, 10, numTest).reshape(-1,1)
    Y_s = np.sin(X_s).flatten()

def runRegression(optimise, initialParams):
    global means, sd
    
    #Create a regressor and run regression.
    regressor = GPRegressor()
    regressor.setJitterFactor(3.0)
    msePreOpt = regressor.runRegression(X.transpose(), Y.transpose(), X_s.transpose(), Y_s.transpose(), initialParams)

    #Optimise.
    if optimise:
        optimiser = GDOptimiser(regressor)
        params = optimiser.optimise(initialParams, 100, 0.001, 0.0001)
        mse = regressor.runRegression(X.transpose(), Y.transpose(), X_s.transpose(), Y_s.transpose(), params)
        print("Predictive MSE without optimisation: %s" % msePreOpt)
        print("Predictive MSE following optimisation: %s" % mse)
    
    means = np.asarray(regressor.getMeans())
    sd = np.asarray(regressor.getStdDev())

def plot():
    pp.plot(X, Y, 'r+')
    pp.plot(X_s, Y_s, 'b-')
    pp.gca().fill_between(X_s.flat, means-sd, means+sd, color="#dddddd")
    pp.plot(X_s, means, 'r--', lw=2)
    pp.title('Mean predictions')
    pp.axis([-10, 10, -1.5, 1.5])
    pp.show()
    
if __name__ == "__main__":
    generateData(1000, 100)
    runRegression(False, {'sigma' : 2.0, 'lambda' : 2.0})
    plot()
