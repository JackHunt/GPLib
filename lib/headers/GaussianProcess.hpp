/*
BSD 3-Clause License

Copyright (c) 2018/19, Jack Miles Hunt
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
#include <optional>
#include <cmath>

#include <CPPUtils/Iterators/CountingIterator.hpp>

#include <Aliases.hpp>
#include <Kernels.hpp>

namespace GPLib {
    template<typename T>
    inline void jitterChol(const Eigen::Ref<const Matrix<T>> A, 
                           Eigen::Ref<Matrix<T>> C) {
        const auto rowsA = A.rows();
        const auto colsA = A.cols();
        assert(rowsA == colsA);

        Matrix<T> jitter = Matrix<T>::Identity(rowsA, colsA);
        jitter *= 1e-8;

        bool passed = false;

        // Successively add jitter to make positive semi-definite.
        while (!passed && jitter(0, 0) < 1e4) {
            Eigen::LLT< Matrix<T> > chol(A + jitter);
            if (chol.info() == Eigen::NumericalIssue) {// Not positive semidefinite.
                jitter *= 1.1;
            }
            else {
                passed = true;
                C = chol.matrixL();
            }
        }

        // If Matrix is still not positive semidefinite.
        if (!passed) {
            throw std::runtime_error("Unable to make matrix positive semidefinite.");
        }
    }

    template<typename T>
    inline void buildCovarianceMatrix(const Eigen::Ref<const Matrix<T>> A, 
                                      const Eigen::Ref<const Matrix<T>> B, 
                                      Eigen::Ref<Matrix<T>> C,
                                      const std::shared_ptr<Kernel<T>> kernel,
                                      const std::optional<const std::string>& gradVar = std::nullopt) {
        const auto rowsA = A.rows();
        const auto rowsB = B.rows();

        auto inner = [&A, &B, &C, &kernel, rowsB, &gradVar](auto i) {
            for (size_t j = i + 1; j < rowsB; j++) {
                if (gradVar.has_value()) {
                    C(i, j) = std::get<T>(kernel->df(A.row(i), B.row(j), gradVar.value()));
                }
                else {
                    C(i, j) = kernel->f(A.row(i), B.row(j));
                }
                C(j, i) = C(i, j);
            }
        };

        CountingIterator<size_t> begin(0);
        CountingIterator<size_t> end(rowsA);
        std::for_each(std::execution::par, begin, end, inner);
    }

    template<typename T>
    inline T logLikelihood(const Eigen::Ref<const Vector<T>> alpha, 
                           const Eigen::Ref<const Matrix<T>> K, 
                           const Eigen::Ref<const Vector<T>> Y) {
        const auto t1 = -0.5 * Y.transpose() * alpha;
        const auto t2 = 0.5 * std::log(K.determinant());
        const auto t3 = (static_cast<T>(K.rows()) / 2) * std::log(2 * M_PI);

        return t1 - t2 - t3;
    }

    template<typename T>
    inline Vector<T> logLikelihoodGrad() {
        return Vector<T>(); // TODO
    }

    template<typename T>
    class GaussianProcess : public std::enable_shared_from_this<GaussianProcess<T>> {
    protected:
        // Covariance Kernel defining this type of regressor.
        std::shared_ptr<Kernel<T>> kernel;

        Matrix<T> alpha;
        Matrix<T> K;

        GaussianProcess(KernelType kernType = KernelType::SQUARED_EXPONENTIAL) {
            switch (kernType) {
            case KernelType::SQUARED_EXPONENTIAL:
                kernel = std::make_shared<SquaredExponential<T>>();
                break;
            default:
                throw std::runtime_error("Invalid kernel choice.");
            }
        }

        virtual void reallocateK(const Eigen::Ref<const Matrix<T>> X, 
                                 const Eigen::Ref<const Vector<T>> Y) {
            // Ensure X and Y contain the same amount of data points.
            assert(X.rows() == Y.rows());

            // Reshape Covariance Matrix and Cholesky Decomposition if required.
            if (K.rows() != X.rows() || K.cols() != X.rows()) {
                K.resize(X.rows(), X.rows());
            }
        }

    public:
        virtual ~GaussianProcess() {
            //
        };

        const std::shared_ptr<const Kernel<T>> getKernel() const {
            return kernel;
        }

        const Matrix<T>& getAlpha() const {
            return alpha;
        }

        const Matrix<T>& getK() const {
            return K;
        }

        virtual void compute(const Eigen::Ref<const Matrix<T>> X, 
                             const Eigen::Ref<const Vector<T>> Y) = 0;

        virtual void train(const Eigen::Ref<const Matrix<T>> X, 
                           const Eigen::Ref<const Vector<T>> Y, 
                           unsigned int maxIterations = 1000) = 0;

        virtual GPOutput<T> predict(const Eigen::Ref<const Matrix<T>> Xs, 
                                    const std::optional<const Eigen::Ref<const Vector<T>>>& Ys = std::nullopt) const = 0;
    };
}

#endif
