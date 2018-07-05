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
        this->buildCovarianceMatrix(X, X, K, kernel);

        // Compute Cholesky Decomposition of K.
        this->jitterChol(K, L);

        // TO-DO
    } while (true);
}

template<typename T>
void GPRegressor<T>::predict(const MapMatrix<T> &Xs, std::optional< const MapVector<T> > &Ys) const {
    // Sanity check ground truth if present.
    if (Ys.has_value()) {
        assert(Xs.rows() == Ys.value().rows());
    }

    // Compute Cross-Covariance Matrix K(X, Xs).
    Matrix<T> Ks(X.rows(), Xs.rows());
    this->buildCovarianceMatrix(X, Xs, Ks, kernel);

    // Solve for Posterior Means.
    const auto tmp = L.triangularView<Eigen::Lower>().solve(Ks);
    const auto posteriorMean = tmp.transpose() * L.triangularView<Eigen::Lower>().solve(Y);
    
    // Compute Posterior Covariance.
    Matrix<T> Kss(Xs.rows(), Xs.rows());
    this->buildCovarianceMatrix(Xs, Xs, Kss, kernel);
    const auto posteriorCov = Kss - tmp.transpose() * tmp;
}

template<typename T>
T GPRegressor<T>::runRegression(const std::vector<T> &trainData, const std::vector<T> &trainTruth,
                                int trainRows, int trainCols, const std::vector<T> &testData, 
                                const std::vector<T> &testTruth, int testRows, int testCols, 
                                const ParameterSet<T> &params) {
    return runRegression(&trainData[0], &trainTruth[0], trainRows, trainCols, &testData[0], 
                         &testTruth[0], testRows, testCols, params);
}

template<typename T>
T GPRegressor<T>::runRegression(const T *trainData, const T *trainTruth, int trainRows,
                                int trainCols, const T *testData, const T *testTruth, 
                                int testRows, int testCols, const ParameterSet<T> &params) {
    /*
    if (trainCols != testCols) {
        throw std::runtime_error("Train and test sets must have the same number of columns.");
    }

    //Wrap data in Eigen matrices and vectors.
    Eigen::Map< const Matrix<T> > X_s(testData, testRows, testCols);
    Eigen::Map< const Vector<T> > Y_s(testTruth, testRows);

    //Reallocate storage, if necessary.
    if (K.rows() != trainRows || K.cols() != trainRows) {
        K.resize(trainRows, trainRows);
        K_s.resize(trainRows, testRows);
        K_ss.resize(testRows, testRows);
    }

    //Compute covariance matrices.
    this->buildCovarianceMatrix(X, X, K, kernel);
    this->buildCovarianceMatrix(X, X_s, K_s, kernel);
    this->buildCovarianceMatrix(X_s, X_s, K_ss, kernel);

    //Add jitter to K.
    if (jitter != 1.0) {
        K += Matrix<T>::Identity(K.rows(), K.cols())*jitter;
    }

    //Solve for alpha.
    Matrix<T> L(trainRows, trainRows);
    jitterChol(K, L);


    //Vector tmp = L.triangularView<Eigen::Lower>().solve(Y);
    //Vector alpha = L.transpose().triangularView<Eigen::Lower>().solve(tmp);

    //Solve for test means and train variances.
    //f_s = K_s.transpose() * alpha;
    //Matrix v = L.triangularView<Eigen::Lower>().solve(K_s);
    //v_s = K_ss - v.transpose() * v;

    const auto tmp = L.triangularView<Eigen::Lower>().solve(K_s);
    f_s = tmp.transpose() * L.triangularView<Eigen::Lower>().solve(Y);
    v_s = K_ss - tmp.transpose() * tmp;

    //Get the MSE.
    auto sq = [](T a) { return a * a; };
    auto predDiff = Y_s - f_s;
    auto predDiffSq = predDiff.unaryExpr(sq);
    return predDiffSq.mean();
    */
    return 0;
}

template<typename T>
std::vector<T> GPRegressor<T>::getMeans() const {
    //return std::vector<T>(f_s.data(), f_s.data() + f_s.rows());
    return {};
}

template<typename T>
std::vector<T> GPRegressor<T>::getCovariances() const {
    //return std::vector<T>(v_s.data(), v_s.data() + v_s.rows());
    return {};
}

template<typename T>
std::vector<T> GPRegressor<T>::getStdDev() const {
    /*
    size_t len = v_s.rows();
    std::vector<T> stdDev(len);

    auto sr = [&stdDev, this](auto iter) {
        const size_t i = iter;
        stdDev[i] = std::sqrt(v_s(i, i)); 
    };

    CountingIterator<size_t> begin(0);
    CountingIterator<size_t> end(len);
    std::for_each(std::execution::par, begin, end, sr);

    return stdDev;
    */
    return {};
}

template<typename T>
void GPRegressor<T>::setJitterFactor(T jitterFactor) {
    jitter = jitterFactor;
}

template class GPRegressor<float>;
template class GPRegressor<double>;
template class GPRegressor<long double>;