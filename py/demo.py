#!/usr/bin/python3

'''
BSD 3-Clause License

Copyright (c) 2017, Jack Miles Hunt
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

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

def runRegression(optimise, initialParams, jitter):
    global means, sd
    
    #Create a regressor and run regression.
    regressor = GPRegressor()
    regressor.setJitterFactor(jitter)
    msePreOpt = regressor.runRegression(X.transpose(), Y.transpose(), X_s.transpose(), Y_s.transpose(), initialParams)
    if not optimise:
        print("Predictive MSE: %s" % msePreOpt)

    #Optimise.
    if optimise:
        optimiser = GDOptimiser()
        optimiser.setJitterFactor(jitter)
        kernel = SquaredExponential()
        params = optimiser.optimise(X.transpose(), Y.transpose(), initialParams, kernel, 100, 0.001, 0.0001)
        mse = regressor.runRegression(X.transpose(), Y.transpose(), X_s.transpose(), Y_s.transpose(), params)
        print("Predictive MSE without optimisation: %s" % msePreOpt)
        print("Predictive MSE following optimisation: %s" % mse)
    
    means = np.asarray(regressor.getMeans())
    sd = np.asarray(regressor.getStdDev())

def plot():
    pp.plot(X, Y, 'r+', label="Ground Truth Training")
    pp.plot(X_s, Y_s, 'b-', label="Ground Truth Test")
    pp.gca().fill_between(X_s.flat, means-sd, means+sd, color="#dddddd")
    pp.plot(X_s, means, 'r--', lw=2, label="Prediction")
    pp.title('Mean predictions')
    pp.axis([-10, 10, -2.0, 2.0])
    pp.legend(loc='upper right')
    pp.show()
    
if __name__ == "__main__":
    generateData(500, 100)
    runRegression(False, {'sigma' : 2.0, 'lambda' : 2.0}, 3.0)
    plot()