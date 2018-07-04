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

#ifndef GPLIB_REGRESSOR_HEADER
#define GPLIB_REGRESSOR_HEADER

#include "GaussianProcess.hpp"

namespace GPLib {
    template<typename T>
    class GPRegressor : GaussianProcess<T> {
    protected:
        //Output predicted mean and covariance.
        Vector<T> f_s;
        Matrix<T> v_s;

        //Covariance matrices.
        Matrix<T> K, K_s, K_ss;

        //Noise to be added to kernel diagonal.
        T jitter = 1.0;

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
        T runRegression(const std::vector<T> &trainData, const std::vector<T> &trainTruth, int trainRows, 
                        int trainCols, const std::vector<T> &testData, const std::vector<T> &testTruth, 
                        int testRows, int testCols, const ParameterSet<T> &params);

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
        T runRegression(const T *trainData, const T *trainTruth, int trainRows, int trainCols,
                        const T *testData, const T *testTruth, int testRows, int testCols,
                        const ParameterSet<T> &params);

        /**
         * @brief getMeans Gets the Means associated with the test set.
         * @return vector of Means.
         */
        std::vector<T> getMeans() const;

        /**
         * @brief getCovariances Gets the Covariance Matrix associated with the test set. GP must be trained first.
         * @return std::vector c-style row major Covariance Matrix.
         */
        std::vector<T> getCovariances() const;

        /**
         * @brief getStdDev Gets Standard Deviations for Means. GP must be trained first.
         * @return vector of Standard Deviations.
         */
        std::vector<T> getStdDev() const;

        /**
         * @brief setJitterFactor Updates jitter factor(noise added to covariance diagonal).
         * @param jitterFactor Jitter(noise) value.
         */
        void setJitterFactor(T jitterFactor);

        /**
         * @brief GPRegressor A Gaussian Process regressor.
         * @param kernType Type of covariance kernel. Defaults to SquaredExponential.
         */
        GPRegressor(KernelType kernType = SQUARED_EXPONENTIAL);
        ~GPRegressor();
    };
}

#endif
