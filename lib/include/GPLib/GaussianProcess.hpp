/*
  BSD 3-Clause License

  Copyright (c) 2024, Jack Miles Hunt
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
#include <optional>
#include <cmath>

#include <CPPUtils/Iterators/CountingIterator.hpp>

#include <GPLib/Aliases.hpp>
#include <GPLib/Kernels.hpp>

namespace GPLib {
  template<typename T>
  inline void jitter_chol(const Eigen::Ref<const Matrix<T>> A,
                          Eigen::Ref<Matrix<T>> C) {
    Matrix<T> jitter = Matrix<T>::Identity(A.rows(), A.cols());
    jitter *= 1e-8;

    bool passed = false;

    // Successively add jitter to make positive semi-definite.
    while (!passed && jitter(0, 0) < 1e4) {
      Eigen::LLT<Matrix<T>> chol(A + jitter);
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
  inline void build_cov(const Eigen::Ref<const Matrix<T>> A,
                        const Eigen::Ref<const Matrix<T>> B,
                        Eigen::Ref<Matrix<T>> C,
                        const std::shared_ptr<const Kernel<T>> kernel,
                        const std::optional<const std::string>& grad_var = std::nullopt) {
    using CPPUtils::Iterators::CountingIterator;

    CountingIterator<decltype(A.rows())> begin(0);
    CountingIterator<decltype(A.rows())> end(A.rows());
    std::for_each(begin, end,
                  [&A, &B, &C, &kernel, &grad_var](auto i) {
                    for (auto j = i + 1; j < B.rows(); j++) {
                      if (grad_var.has_value()) {
                        C(i, j) = std::get<T>(kernel->df(A.row(i), B.row(j), grad_var.value()));
                      }
                      else {
                        C(i, j) = kernel->f(A.row(i), B.row(j));
                      }
                      C(j, i) = C(i, j);
                    }
                  });
  }

  template<typename T>
  inline T log_likelihood(const Eigen::Ref<const Vector<T>> alpha,
                          const Eigen::Ref<const Matrix<T>> K,
                          const Eigen::Ref<const Vector<T>> Y) {
    const auto t1 = -0.5 * Y.transpose() * alpha;
    const auto t2 = 0.5 * std::log(K.determinant());
    const auto t3 = (static_cast<T>(K.rows()) / 2) * std::log(2 * M_PI);

    return t1 - t2 - t3;
  }

  template<typename T>
  inline Vector<T> log_likelihood_grad() {
    return Vector<T>(); // TODO
  }

  template<typename T>
  class GaussianProcess : public std::enable_shared_from_this<GaussianProcess<T>> {
  protected:
    // Covariance Kernel defining this type of regressor.
    std::shared_ptr<Kernel<T>> kernel;

    // Copy of training data X and ground truth Y.
    Matrix<T> X;
    Vector<T> Y;

    // GP temporaries.
    Matrix<T> alpha;
    Matrix<T> K;
    Matrix<T> L;

    GaussianProcess(KernelType kernType = KernelType::SQUARED_EXPONENTIAL) {
      switch (kernType) {
      case KernelType::SQUARED_EXPONENTIAL:
        kernel = std::make_shared<SquaredExponential<T>>();
        break;
      default:
        throw std::runtime_error("Invalid kernel choice.");
      }
    }

    virtual void reallocate(const Eigen::Ref<const Matrix<T>> X,
                            const Eigen::Ref<const Vector<T>> Y) {
      // Ensure X and Y contain the same amount of data points.
      assert(X.rows() == Y.rows());

      // Reshape Covariance Matrix and Cholesky Decomposition if required.
      if (K.rows() != X.rows() || K.cols() != X.rows()) {
        K.resize(X.rows(), X.rows());
        L.resize(K.rows(), K.rows());
      }
    }

  public:
    virtual ~GaussianProcess() {
      //
    };

    std::shared_ptr<const Kernel<T>> getKernel() const {
      return kernel;
    }

    std::shared_ptr<Kernel<T>> getKernel() {
      return kernel;
    }

    Eigen::Ref<const Matrix<T>> getAlpha() const {
      return alpha;
    }

    Eigen::Ref<const Matrix<T>> getK() const {
      return K;
    }

    Eigen::Ref<const Matrix<T>> getX() const {
      return X;
    }

    Eigen::Ref<const Vector<T>> getY() const {
      return Y;
    }

    virtual void compute(const Eigen::Ref<const Matrix<T>> X,
                         const Eigen::Ref<const Vector<T>> Y) = 0;

    virtual GPOutput<T> predict(const Eigen::Ref<const Matrix<T>> Xs,
                                const std::optional<const Eigen::Ref<const Vector<T>>>& Ys = std::nullopt) const = 0;
  };
}

#endif
