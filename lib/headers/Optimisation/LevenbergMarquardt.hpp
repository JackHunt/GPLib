/*
BSD 3-Clause License

Copyright (c) 2020, Jack Miles Hunt
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

#ifndef GPLIB_LEVENBERG_MARQUARDT_HEADER
#define GPLIB_LEVENBERG_MARQUARDT_HEADER

#include "Optimiser.hpp"

namespace GPLib::Optimisation {
    template<typename T>
    class LMParameters : public OptimiserParameters<T> {
    protected:
        T lambda;

    public:
        LMParameters(std::shared_ptr<GaussianProcess<T>> gp,
                     const MappedMatrix<T>& X,
                     const MappedVector<T>& Y,
                     T lambda = 0.1,
                     unsigned int maxIterations = 100,
                     T minConvergenceNorm = 1e-3,
                     unsigned int convergenceWindow = 5) :
            OptimiserParameters(X, Y, maxIterations, 
                                minConvergenceNorm, 
                                convergenceWindow) {
            // Verify lambda.
            assert(lambda > 0);
        }

        T getLambda() const {
            return lambda;
        }

        void setLambda(T lambda) {
            assert(lambda > 0);
            this->lambda = lambda;
        }
    };

    template<typename T>
    class LevenbergMarquardt : public Optimiser<T> {
    protected:
        T lambda;
        Matrix<T> gradK;

    public:
        LevenbergMarquardt(const LMParameters<T>& parameters);
        virtual ~LevenbergMarquardt();

        virtual void operator()() override;
    };
}

#endif