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

void GPRegressor::runRegression(const double *trainData, const double *trainTruth, int trainRows, int trainCols,
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
	//Eigen::MatrixXf K(1,1);
	//Eigen::MatrixXf K_s(1,1);
	//Eigen::MatrixXf K_ss(1,1);
    //buildCovarianceMatrix(X, X, K, params);
	//buildCovarianceMatrix(X, X_s, K_s, params);
	//buildCovarianceMatrix(X_s, X_s, K_ss, params);

	//Solve for alpha.

	//Solve for solutions.
}

void GPRegressor::buildCovarianceMatrix(const Eigen::Map<Matrix> &A, const Eigen::Map<Matrix> &B,
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

void GPRegressor::jitterChol(const Eigen::Map<Matrix> &A, Matrix &C){
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
