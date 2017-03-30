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

double GPRegressor::runRegression(const double *trainData, const double *trainTruth, int trainRows, int trainCols,
								  const double *testData, const double *testTruth, int testRows, int testCols,
								  const ParamaterSet &params){
	if(trainCols != testCols){
		throw std::runtime_error("Train and test sets must have the same number of columns.");
	}
	
	//Wrap data in Eigen matrices and vectors.
	const Eigen::Map<const Matrix> X(trainData, trainRows, trainCols);
	const Eigen::Map<const Matrix> X_s(testData, testRows, testCols);
	const Eigen::Map<const Vector> Y(trainTruth, trainRows);
	const Eigen::Map<const Vector> Y_s(testTruth, testRows);

	//Compute covariance matrices.
	Matrix K(trainRows, trainRows);
	Matrix K_s(trainRows, testRows);
	Matrix K_ss(testRows, testRows);
    buildCovarianceMatrix(X, X, K, params);
	buildCovarianceMatrix(X, X_s, K_s, params);
	buildCovarianceMatrix(X_s, X_s, K_ss, params);

	//Solve for alpha.
	Matrix L(trainRows, trainRows);
	jitterChol(X, L);
	Vector tmp = L.triangularView<Eigen::Lower>().solve(Y);
	Vector alpha = L.transpose().triangularView<Eigen::Lower>().solve(tmp);

	//Solve for solutions.
	Vector f_s = K_s.transpose() * alpha;
	Vector v = L.triangularView<Eigen::Lower>().solve(K_s);
	Matrix v_s = K_ss - v.transpose() * v;
	
}

void GPRegressor::buildCovarianceMatrix(const Eigen::Map<const Matrix> &A, const Eigen::Map<const Matrix> &B,
										Matrix &C, const ParamaterSet &params){
	if(A.cols() != B.cols() || A.cols() != C.cols()){
		throw std::runtime_error("Matrix column dimensions must match when generating covariances.");
	}

	for(int i = 0; i < A.rows(); i++){
		for(int j = 0; j < B.rows(); j++){
			C(i, j) = kernel->f(A.row(i), B.row(j), params);
		}
	}
}

void GPRegressor::jitterChol(const Eigen::Map<const Matrix> &A, Matrix &C){
	Matrix jitter = Matrix::Identity(A.rows(), A.cols());
	jitter *= 1e-8;
	
	bool passed = false;

	while(!passed && jitter(0,0) < 1e4){
		Eigen::LLT<Matrix> chol(A + jitter);
		if(chol.info() == Eigen::NumericalIssue){
			jitter *= 1.1;
		}else{
			passed = true;
			C = chol.matrixL();
		}
	}

	if(!passed){
		throw std::runtime_error("Unable to make matrix positive semidefinite.");
	}
}
