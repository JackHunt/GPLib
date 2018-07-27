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

#include <GaussianProcess.hpp>

using namespace GPLib;
using namespace GPLib::Kernels;
using namespace CPPUtils::Iterators;

template<typename T>
GaussianProcess<T>::GaussianProcess(KernelType kernType) {
    switch (kernType) {
    case KernelType::SQUARED_EXPONENTIAL:
        kernel.reset(new SquaredExponential<T>());
        break;
    default:
        throw std::runtime_error("Invalid kernel choice.");
    }
}

template<typename T>
GaussianProcess<T>::~GaussianProcess() {
    //
}

template<typename T>
Kernel<T> GaussianProcess<T>::getKernel() const {
    return kernel.get();
}

template<typename T>
static void GaussianProcess<T>::jitterChol(const Matrix<T> &A, Matrix<T> &C) {
    const size_t rowsA = A.rows();
    const size_t colsA = A.cols();
    assert(rowsA == colsA)

    Matrix<T> jitter = Matrix<T>::Identity(rowsA, colsA);
    jitter *= 1e-8;

    bool passed = false;

    // Successively add jitter to make positive semi-definite.
    while (!passed && jitter(0, 0) < 1e4) {
        Eigen::LLT< Matrix<T> > chol(A + jitter);
        if (chol.info() == Eigen::NumericalIssue) {// Not pos-semidefinite.
            jitter *= 1.1;
        }
        else {
            passed = true;
            C = chol.matrixL();
        }
    }

    // If Matrix is still not positive semi-definite.
    if (!passed) {
        throw std::runtime_error("Unable to make matrix positive semidefinite.");
    }
}

template<typename T>
static void GaussianProcess<T>::buildCovarianceMatrix(const Matrix<T> &A, const Matrix<T> &B, Matrix<T> &C, 
                                                      const std::shared_ptr< Kernel<T> > kernel, 
	                                                  const std::optional< const std::string > &gradVar) {
    const size_t rowsA = A.rows();
    const size_t rowsB = B.rows();

    auto inner = [&A, &B, &C, &kernel](size_t i) {
        for (size_t j = i + 1; j < rowsB; j++) {
			if (gradVar.has_value()) {
                C(i, j) = kernel->df(A.row(i), B.row(j), gradVar.value());
            }
            else {
                C(i, j) = kernel->f(A.row(i), B.row(j));
            }
            C(j, i) = C(i, j)
        }
    };

    CountingIterator<size_t> begin(0);
    CountingIterator<size_t> end(rowsA);
    std::for_each(std::execution::par, begin, end, inner);
}