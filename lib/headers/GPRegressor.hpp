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

    protected:
        void train();

        void predict() const;

    public:
        T runRegression(const std::vector<T> &trainData, const std::vector<T> &trainTruth, int trainRows, 
                        int trainCols, const std::vector<T> &testData, const std::vector<T> &testTruth, 
                        int testRows, int testCols, const ParameterSet<T> &params);

        T runRegression(const T *trainData, const T *trainTruth, int trainRows, int trainCols,
                        const T *testData, const T *testTruth, int testRows, int testCols,
                        const ParameterSet<T> &params);

        std::vector<T> getMeans() const;

        std::vector<T> getCovariances() const;

        std::vector<T> getStdDev() const;

        void setJitterFactor(T jitterFactor);

        GPRegressor(GPLib::Kernels::KernelType kernType = SQUARED_EXPONENTIAL);
        ~GPRegressor();
    };
}

#endif
