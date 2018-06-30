/*
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
*/

#ifndef GP_REGRESSOR_HEADER
#define GP_REGRESSOR_HEADER

#include "Kernels.hpp"
#include "typedefs.hpp"
#include "Util.hpp"

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <iostream>

namespace GPLib {
	class GPRegressor{
    private:
        //Kernel defining this type of regressor.
		std::shared_ptr<Kernel> kernel;

        //Output predicted mean and covariance.
		Vector f_s;
		Matrix v_s;

        //Covariance matrices.
		Matrix K, K_s, K_ss;

        //Noise to be added to kernel diagonal.
		double jitter = 1.0;
		
	public:
        /**
         * @brief runRegression Trains a Gaussian Process regression model.
         * @param trainData Training data in row major c-style array format.
         * @param trainTruth Training ground truth vector.
         * @param trainRows # training rows.
         * @param trainCols # training cols(data dimensionality).
         * @param testData Testing(prediction) data in row major c-style array format.
         * @param testTruth Testing ground truth vector.
         * @param testRows # testing rows.
         * @param testCols # testing cols(data dimensionality)
         * @param params Parameters for kernel(must match kernel type of this GP).
         * @return Mean Squared Error between predicted means and ground truth for test set.
         */
		double runRegression(const std::vector<double> &trainData, const std::vector<double> &trainTruth, int trainRows, int trainCols,
							 const std::vector<double> &testData, const std::vector<double> &testTruth, int testRows, int testCols,
                             const ParameterSet &params);

        /**
         * @brief runRegression Trains a Gaussian Process regression model.
         * @param trainData Training data in row major c-style array format.
         * @param trainTruth Training ground truth vector.
         * @param trainRows # training rows.
         * @param trainCols # training cols(data dimensionality).
         * @param testData Testing(prediction) data in row major c-style array format.
         * @param testTruth Testing ground truth vector.
         * @param testRows # testing rows.
         * @param testCols # testing cols(data dimensionality)
         * @param params Parameters for kernel(must match kernel type of this GP).
         * @return Mean Squared Error between predicted means and ground truth for test set.
         */
		double runRegression(const double *trainData, const double *trainTruth, int trainRows, int trainCols,
							 const double *testData, const double *testTruth, int testRows, int testCols,
                             const ParameterSet &params);

//#ifdef WITH_PYTHON_BINDINGS
        /**
         * @brief runRegression Trains a Gaussian Process regression model.
         * @param trainData Training data in row major c-style array format.
         * @param trainCols # training cols(data dimensionality).
         * @param trainRows # training rows.
         * @param trainTruth Training ground truth vector.
         * @param trainTruthRows # training rows
         * @param testData Testing(prediction) data in row major c-style array format.
         * @param testCols # testing cols(data dimensionality)
         * @param testRows # testing rows.
         * @param testTruth Testing ground truth vector.
         * @param testTruthRows # testing rows.
         * @param params Parameters for kernel(must match kernel type of this GP).
         * @return Mean Squared Error between predicted means and ground truth for test set.
         */
		double runRegression(const double *trainData, int trainCols, int trainRows, const double *trainTruth,
							 int trainTruthRows, const double *testData, int testCols, int testRows,
                             const double *testTruth, int testTruthRows, const ParameterSet &params);
//#endif

        /**
         * @brief getMeans Gets the Means associated with the test set.
         * @return vector of Means.
         */
		std::vector<double> getMeans() const;

        /**
         * @brief getCovariances Gets the Covariance Matrix associated with the test set. GP must be trained first.
         * @return std::vector c-style row major Covariance Matrix.
         */
		std::vector<double> getCovariances() const;

        /**
         * @brief getStdDev Gets Standard Deviations for Means. GP must be trained first.
         * @return vector of Standard Deviations.
         */
        std::vector<double> getStdDev();

       /**
        * @brief setJitterFactor Updates jitter factor(noise added to covariance diagonal).
        * @param jitterFactor Jitter(noise) value.
        */
		void setJitterFactor(double jitterFactor);
		
        /**
         * @brief GPRegressor A Gaussian Process regressor.
         * @param kernType Type of covariance kernel. Defaults to SquaredExponential.
         */
		GPRegressor(KernelType kernType = SQUARED_EXPONENTIAL);
        ~GPRegressor();
	};
}

#endif
