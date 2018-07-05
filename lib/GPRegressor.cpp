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

#include <GPRegressor.hpp>

using namespace GPLib;
using namespace GPLib::Kernels;
using namespace CPPUtils::Iterators;

template<typename T>
GPRegressor<T>::GPRegressor(KernelType kernType) : GaussianProcess<T>(kernType) {
    //
}

template<typename T>
GPRegressor<T>::~GPRegressor() {
    //
}

template<typename T>
void GPRegressor<T>::train(const MapMatrix<T> &XMap, const MapVector<T> &YMap) {
    // Sanity check.
    assert(XMap.rows() == YMap.rows());

    // Reshape Covariance Matrix and Cholesky Decomposition if required.
    if (K.rows() != XMap.rows() || K.cols() != XMap.rows()) {
        K.resize(XMap.rows(), XMap.rows());
        L.resize(XMap.rows(), XMap.rows());
    }

    // Make a copy of XMap and YMap.
    if (XMap.rows() != X.rows() || XMap.cols() != X.cols()) {
        X.resize(XMap.rows(), XMap.cols());
    }
    X = XMap;

    if (YMap.rows() != Y.rows()) {
        Y.resize(YMap.rows(), 1);
    }
    Y = YMap;

    // Optimise Log Marginal Likelihood with Levenberg-Marquardt.
    T lambda = 1.0;
    do {
        // Compute Covariance Matrix K(X, X^t).
        buildCovarianceMatrix(X, X, K, kernel);

        // Compute Cholesky Decomposition of K.
        jitterChol(K, L);

        // Compute Alpha.
        const auto alpha = L.triangularView<Eigen::Lower>().solve(Y);

        // Compute gradient of GP w.r.t. K.
        const auto dfdk = alpha * alpha.transpose() - K.inverse();

        // Compute gradient of K.
        // TODO
    } while (true);
}

template<typename T>
GPOutput<T> GPRegressor<T>::predict(const MapMatrix<T> &Xs, std::optional< const MapVector<T> > &Ys) const {
    // Sanity check ground truth if present.
    if (Ys.has_value()) {
        assert(Xs.rows() == Ys.value().rows());
    }

    // Compute Cross-Covariance Matrix K(X, Xs).
    Matrix<T> Ks(X.rows(), Xs.rows());
    buildCovarianceMatrix(X, Xs, Ks, kernel);

    // Solve for Posterior Means.
    const auto tmp = L.triangularView<Eigen::Lower>().solve(Ks);
    const auto posteriorMean = tmp.transpose() * L.triangularView<Eigen::Lower>().solve(Y);
    
    // Compute Posterior Covariance.
    Matrix<T> Kss(Xs.rows(), Xs.rows());
    buildCovarianceMatrix(Xs, Xs, Kss, kernel);
    const auto posteriorCov = Kss - tmp.transpose() * tmp;

    // Return Mean and Covariance if no ground truth.
    if (!Ys.has_value()) {
        return GPOutput<T>(MeanCov<T>(posteriorMean, posteriorCov));
    }

    // Otherwise, return Mean, Covariance and MSE.
    const auto predDiff = Ys.value() - posteriorMean;
    const T mse = predDiff.unaryExpr([](T a) { return a * a; }).mean();
    return GPOutput<T>(MeanCovErr<T>(posteriorMean, posteriorCov, mse));
}

template<typename T>
T GPRegressor<T>::logLikelihood(const Vector<T> &alpha, const Matrix<T> &K, const Vector<T> &Y) const {
    const T t1 = -0.5 * Y.transpose() * alpha;
    const T t2 = 0.5 * std::log(K.determinant());
    const T t3 = (static_cast<T>(K.rows()) / 2.0) * std::log(2.0 * M_PI);

    return t1 - t2 - t3;
}

template<typename T>
Vector<T> GPRegressor<T>::logLikelihoodGrad() const {
    return Vector<T>();
}

template class GPRegressor<float>;
template class GPRegressor<double>;
template class GPRegressor<long double>;