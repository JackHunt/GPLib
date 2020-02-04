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

#include <GPRegressor.hpp>

using namespace GPLib;
using namespace CPPUtils::Iterators;

template<typename T>
GPRegressor<T>::GPRegressor(KernelType kernType) {
    //
}

template<typename T>
GPRegressor<T>::~GPRegressor() {
    //
}

template<typename T>
void GPRegressor<T>::compute(const Eigen::Ref<const Matrix<T>> X,
                             const Eigen::Ref<const Vector<T>> Y) {
    // Ensure K is the correct dimensionality.
    reallocate(X, Y);

    // Compute Covariance Matrix K(X, X^t).
    buildCovarianceMatrix<T>(X, X.transpose(), K, kernel);

    // Compute the Cholesky Decomposition of K.
    Matrix<T> chol(X.rows(), X.rows());
    jitterChol<T>(K, chol);

    // Compute alpha.
    alpha = std::move(chol.triangularView<Eigen::Lower>().solve(Y));
}

template<typename T>
void GPRegressor<T>::train(const Eigen::Ref<const Matrix<T>> X, 
                           const Eigen::Ref<const Vector<T>> Y, 
                           unsigned int maxIterations) {
    /*
    // Current parameters.
    auto params = kernel->getParameters();
    
    // Temporary storage.
    Matrix<T> gradK, gradL;
    Vector<T> paramVec, nabla;

    paramVec.resize(params.size(), 1);
    ;

    // Identity for step computation.
    

    // Optimise Log Marginal Likelihood with Levenberg-Marquardt.
    T lambda = 1.0;
    size_t epoch = 0;
    while (epoch <= maxIterations) {



            // Recompute loss with new params.
        buildCovarianceMatrix<T>(X, X.transpose(), K, gp->getKernel());
        jitterChol(K, L);
        const auto newAlpha = L.triangularView<Eigen::Lower>().solve(Y);
        const T updatedLogLik = logLikelihood(newAlpha, K, Y);
        
        // Update lambda and reject or accept change.
        if (updatedLogLik >= logLik) {
            lambda *= 10.0;
        }
        else {
            lambda /= 10.0;
            //Update params.
            size_t idx = 0;
            for (auto& p : params) {
                p.second = updatedParams(idx);
                idx++;
            }
            kernel->setParameters(params);
        }
        epoch++;
    }
    */
}

template<typename T>
GPOutput<T> GPRegressor<T>::predict(const Eigen::Ref<const Matrix<T>> Xs, 
                                    const std::optional<const Eigen::Ref<const Vector<T>>>& Ys) const {
    // Sanity check ground truth if present.
    if (Ys.has_value()) {
        assert(Xs.rows() == Ys.value().rows());
    }

    // Compute Cross-Covariance Matrix K(X, Xs).
    Matrix<T> Ks(X.rows(), Xs.rows());
    buildCovarianceMatrix<T>(X, Xs, Ks, kernel);

    // Solve for Posterior Means.
    const auto tmp = L.triangularView<Eigen::Lower>().solve(Ks);
    const auto posteriorMean = tmp.transpose() * L.triangularView<Eigen::Lower>().solve(Y);
    
    // Compute Posterior Covariance.
    Matrix<T> Kss(Xs.rows(), Xs.rows());
    buildCovarianceMatrix<T>(Xs, Xs.transpose(), Kss, kernel);
    const auto posteriorCov = Kss - tmp.transpose() * tmp;

    // Return Mean and Covariance if no ground truth is provided.
    if (!Ys.has_value()) {
        return GPOutput<T>(MeanCov<T>(posteriorMean, posteriorCov));
    }

    // Otherwise, return Mean, Covariance and MSE.
    const auto predDiff = Ys.value() - posteriorMean;
    const auto mse = predDiff.unaryExpr([](auto a) {
        return a * a; 
    }).mean();
    
    return GPOutput<T>(MeanCovErr<T>(posteriorMean, posteriorCov, mse));
}

template class GPRegressor<float>;
template class GPRegressor<double>;
template class GPRegressor<long double>;