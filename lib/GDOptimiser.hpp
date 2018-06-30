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

#ifndef GPLIB_GD_OPTIMISER_HEADER
#define GPLIB_GD_OPTIMISER_HEADER

#include "GPRegressor.hpp"
#include "Util.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace GPLib {
    template<typename T>
    class GDOptimiser {
    private:
        //Noise to be added to covariance diagonal.
        T jitter = 1.0;

        //Covariance matrix, it's derivative and cholesky factorisation.
        Matrix<T> K, K_deriv, K_chol;

        /**
         * @brief logMarginalLikelihood Log marginal likelihood given current Parameters.
         * @param alpha Alpha - as per GPML
         * @param K Covariance matrix.
         * @param Y Ground truth vector.
         * @param rows # rows in cov/ground truth.
         * @return Marginal Log Likelihood - as per GPML.
         */
        T logMarginalLikelihood(const Vector<T> &alpha, const Matrix<T> &K, const Vector<T> &Y, int rows) const;

    public:
        /**
         * @brief GDOptimiser
         */
        GDOptimiser();

        /**
         * @brief ~GDOptimiser
         */
        ~GDOptimiser();

        /**
         * @brief optimise Maximises log-marginal-likelihood as per GPML.
         * @param trainData Training data in row major c-style array format.
         * @param trainTruth Training ground truth vector.
         * @param trainRows # training rows.
         * @param trainCols # training cols(data dimensionality).
         * @param params Initial Parameter struct(must match kernel choice).
         * @param kernel Covariance kernel(must match Parameter struct).
         * @param iterations Max # gradient iterations.
         * @param targetStepSize Mininum step size for termination.
         * @param learnRate Gradient update multiplier.
         * @return Optimised Parameters.
         */
        ParameterSet<T> optimise(const std::vector<T> &trainData, const std::vector<T> &trainTruth, int trainRows,
                                 int trainCols, const ParameterSet<T> &params, const std::shared_ptr< Kernel<T> > &kernel, 
                                 int iterations, T targetStepSize, T learnRate);

        /**
         * @brief optimise Maximises log-marginal-likelihood as per GPML.
         * @param trainData Training data in row major c-style array format.
         * @param trainTruth Training ground truth vector.
         * @param trainRows # training rows.
         * @param trainCols # training cols(data dimensionality).
         * @param params Initial Parameter struct(must match kernel choice).
         * @param kernel Covariance kernel(must match Parameter struct).
         * @param iterations Max # gradient iterations.
         * @param targetStepSize Mininum step size for termination.
         * @param learnRate Gradient update multiplier.
         * @return Optimised Parameters.
         */
        ParameterSet<T> optimise(const T *trainData, const T *trainTruth, int trainRows, int trainCols,
                                 const ParameterSet<T> &params, const std::shared_ptr< Kernel<T> > &kernel, int iterations,
                                 T targetStepSize, T learnRate);

        /**
         * @brief setJitterFactor Updates jitter factor(noise added to covariance diagonal).
         * @param jitterFactor Jitter(noise) value.
         */
        void setJitterFactor(T jitterFactor);
    };
}

#endif
