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

#ifndef GPLIB_GAUSSIAN_PROCESS_HEADER
#define GPLIB_GAUSSIAN_PROCESS_HEADER

#include <vector>
#include <iostream>
#include <memory>
#include <algorithm>
#include <execution>

#include <CPPUtils/Iterators/CountingIterator.hpp>

#include <Aliases.hpp>
#include <Kernels.hpp>

namespace GPLib {
    template<typename T>
    class GaussianProcess {
    protected:
        static void jitterChol(const Matrix<T> &A, Matrix<T> &C);
        
        static void buildCovarianceMatrix(const MapMatrix<T> &A, const MapMatrix<T> &B, Matrix<T> &C, 
                                          const std::shared_ptr< GPLib::Kernels::Kernel<T> > kernel);

        virtual T logLikelihood() = 0;

        virtual Vector<T> logLikelihoodGrad() = 0;

    protected:
        // Covariance Kernel defining this type of regressor.
        std::shared_ptr< GPLib::Kernels::Kernel<T> > kernel;

        //Noise to be added to kernel diagonal.
        T jitter = 1.0;

    public:
        GaussianProcess(GPLib::Kernels::KernelType kernType);

        virtual ~GaussianProcess();

        GPLib::Kernels::Kernel<T> getKernel() const;

        virtual void train() = 0;

        virtual void predict() const = 0;
    };
}

#endif
