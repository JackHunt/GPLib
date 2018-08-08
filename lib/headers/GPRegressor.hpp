/*
BSD 3-Clause License

Copyright (c) 2018, Jack Miles Hunt
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

#include <GaussianProcess.hpp>

namespace GPLib {
    template<typename T>
    class GPRegressor : GaussianProcess<T> {
    protected:
        // Copy of training data X and ground truth Y.
        Matrix<T> X;
        Vector<T> Y;

        // Covariance Matrix K(X, X^t) and it's Cholesky Decomposition.
        Matrix<T> K, L;

    protected:
        T logLikelihood(const Vector<T> &alpha, const Matrix<T> &K, const Vector<T> &Y) const;

        Vector<T> logLikelihoodGrad() const;

    public:
        void train(const MapMatrix<T> &X, const MapVector<T> &Y, size_t maxEpochs);

        GPOutput<T> predict(const MapMatrix<T> &Xs, const std::optional< const MapVector<T> > &Ys) const;

        GPRegressor(GPLib::Kernels::KernelType kernType = GPLib::Kernels::KernelType::SQUARED_EXPONENTIAL);
        virtual ~GPRegressor();
    };
}

#endif
