#include "GPRegressor.h"

using namespace GaussianProcess;

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
								  const ParamaterSet &params){
	runRegression(&trainData[0], &trainTruth[0], trainRows, trainCols, &testData[0], &testTruth[0], testRows, testCols, params);
}

//#ifdef WITH_PYTHON_BINDINGS
double GPRegressor::runRegression(const double *trainData, int trainCols, int trainRows, const double *trainTruth,
								  int trainTruthRows, const double *testData, int testCols, int testRows,
								  const double *testTruth, int testTruthRows, const ParamaterSet &params){
	runRegression(trainData, trainTruth, trainRows, trainCols, testData, testTruth, testRows, testCols, params);
}
//#endif

double GPRegressor::runRegression(const double *trainData, const double *trainTruth, int trainRows, int trainCols,
								  const double *testData, const double *testTruth, int testRows, int testCols,
								  const ParamaterSet &params){
	if(trainCols != testCols){
		throw std::runtime_error("Train and test sets must have the same number of columns.");
	}
	
	//Wrap data in Eigen matrices and vectors.
	Eigen::Map<const Matrix> X(trainData, trainRows, trainCols);
	Eigen::Map<const Matrix> X_s(testData, testRows, testCols);
	Eigen::Map<const Vector> Y(trainTruth, trainRows);
	Eigen::Map<const Vector> Y_s(testTruth, testRows);
	Y_copy = Vector(Y);
	X_copy = Matrix(X);

	//Compute covariance matrices.
    Matrix K(trainRows, trainRows);
	Matrix K_s(trainRows, testRows);
	Matrix K_ss(testRows, testRows);
    buildCovarianceMatrix< Eigen::Map<const Matrix> >(X, X, K, params, kernel);
	buildCovarianceMatrix< Eigen::Map<const Matrix> >(X, X_s, K_s, params, kernel);
	buildCovarianceMatrix< Eigen::Map<const Matrix> >(X_s, X_s, K_ss, params, kernel);

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
#pragma omp parallel for schedule(dynamic)
#endif
	for(size_t i = 0; i < len; i++) {
		stdDev[i] = sqrt(v_s(i, i));
	}

	return stdDev;
}

void GPRegressor::setJitterFactor(double jitterFactor) {
	jitter = jitterFactor;
}
