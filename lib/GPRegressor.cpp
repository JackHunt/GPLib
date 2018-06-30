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

#include "GPRegressor.hpp"

using namespace GPLib;

GPRegressor::GPRegressor(KernelType kernType){
	switch(kernType){
	case SQUARED_EXPONENTIAL:
		kernel.reset(new SquaredExponential());
		break;
	default:
		throw std::runtime_error("Invalid kernel choice.");
	}
}

GPRegressor::~GPRegressor(){
	//
}

double GPRegressor::runRegression(const std::vector<double> &trainData, const std::vector<double> &trainTruth, int trainRows, int trainCols,
								  const std::vector<double> &testData, const std::vector<double> &testTruth, int testRows, int testCols,
                                  const ParameterSet &params){
	return runRegression(&trainData[0], &trainTruth[0], trainRows, trainCols, &testData[0], &testTruth[0], testRows, testCols, params);
}

double GPRegressor::runRegression(const double *trainData, const double *trainTruth, int trainRows, int trainCols,
								  const double *testData, const double *testTruth, int testRows, int testCols,
                                  const ParameterSet &params){
	if(trainCols != testCols){
		throw std::runtime_error("Train and test sets must have the same number of columns.");
	}
	
	//Wrap data in Eigen matrices and vectors.
	Eigen::Map<const Matrix> X(trainData, trainRows, trainCols);
	Eigen::Map<const Matrix> X_s(testData, testRows, testCols);
	Eigen::Map<const Vector> Y(trainTruth, trainRows);
	Eigen::Map<const Vector> Y_s(testTruth, testRows);

		//Reallocate storage, if necessary.
	if(K.rows() != trainRows || K.cols() != trainRows) {
		K.resize(trainRows, trainRows);
		K_s.resize(trainRows, testRows);
	    K_ss.resize(testRows, testRows);
	}
	
	//Compute covariance matrices.
    buildCovarianceMatrix(X, X, K, params, kernel);
	buildCovarianceMatrix(X, X_s, K_s, params, kernel);
	buildCovarianceMatrix(X_s, X_s, K_ss, params, kernel);

	//Add jitter to K.
	if(jitter != 1.0) {
		K += Matrix::Identity(K.rows(), K.cols())*jitter;
	}
	
	//Solve for alpha.
	Matrix L(trainRows, trainRows);
	jitterChol(K, L);

	/*
	Vector tmp = L.triangularView<Eigen::Lower>().solve(Y);
	Vector alpha = L.transpose().triangularView<Eigen::Lower>().solve(tmp);

	//Solve for test means and train variances.
	f_s = K_s.transpose() * alpha;
	Matrix v = L.triangularView<Eigen::Lower>().solve(K_s);
	v_s = K_ss - v.transpose() * v;
	*/

	Matrix tmp = L.triangularView<Eigen::Lower>().solve(K_s);
	f_s = tmp.transpose() * L.triangularView<Eigen::Lower>().solve(Y);
	v_s = K_ss - tmp.transpose() * tmp;

	//Get the MSE.
	auto sq = [](double a){return a*a;};
	Vector predDiff = Y_s - f_s;
	predDiff = predDiff.unaryExpr(sq);
	return predDiff.mean();
}

std::vector<double> GPRegressor::getMeans() const {
	return std::vector<double>(f_s.data(), f_s.data() + f_s.rows());
}

std::vector<double> GPRegressor::getCovariances() const {
	return std::vector<double>(v_s.data(), v_s.data() + v_s.rows());
}

std::vector<double> GPRegressor::getStdDev() {
	const size_t len = v_s.rows();
	std::vector<double> stdDev(len);
#ifdef WITH_OPENMP
#pragma omp parallel for
#endif
	for(int i = 0; i < len; i++) {
		stdDev[i] = sqrt(v_s(i, i));
	}

	return stdDev;
}

void GPRegressor::setJitterFactor(double jitterFactor) {
	jitter = jitterFactor;
}
