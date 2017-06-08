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

def runRegression(optimise):
    global means, sd
    
    #Create a regressor and run regression.
    regressor = GPRegressor()
    regressor.setJitterFactor(4.0)
    mse = regressor.runRegression(X.transpose(), Y.transpose(), X_s.transpose(), Y_s.transpose(), {'sigma' : 2.0, 'lambda' : 2.0})

    #Optimise.
    if optimise:
        pass
    
    means = np.asarray(regressor.getMeans())
    sd = np.asarray(regressor.getStdDev())
    
    #Give some feedback.
    print("Predictive MSE: %s" % mse)

def plot():
    pp.plot(X, Y, 'r+')
    pp.plot(X_s, Y_s, 'b-')
    pp.gca().fill_between(X_s.flat, means-2*sd, means+2*sd, color="#dddddd")
    pp.plot(X_s, means, 'r--', lw=2)
    pp.title('Mean predictions')
    pp.axis([-10, 10, -1.5, 1.5])
    pp.show()
    
if __name__ == "__main__":
    generateData(100, 100)
    runRegression(True)
    plot()
