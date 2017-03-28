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

void GPRegressor::runRegression(const float *trainData, const float *trainTruth, int trainRows, int trainCols,
								const float *testData, const float testTruth, int testRows, int testCols,
								const ParamaterSet &params){
	if(trainCols != testCols){
		throw std::runtime_error("Train and test sets must have the same number of columns.");
	}
	
	//Wrap data in Eigen matrices.

	//Compute covariance matrices.
	computeCovarianceMatrices(params);

	//Solve for alpha.

	//Solve for solutions.
}

void GPRegressor::computeCovarianceMatrices(const ParamaterSet &params){
	buildCovarianceMatrix(X, X, K, params);
	buildCovarianceMatrix(X, X_s, K_s, params);
	buildCovarianceMatrix(X_s, X_s, K_ss, params);
}

void GPRegressor::buildCovarianceMatrix(const std::shared_ptr<Eigen::Map<Eigen::MatrixXf> > &A,
										const std::shared_ptr<Eigen::Map<Eigen::MatrixXf> > &B,
										std::shared_ptr<Eigen::MatrixXf> &C, const ParamaterSet &params){
	if(A->cols() != B->cols() || A->cols() != C->cols()){
		throw std::runtime_error("Matrix column dimensions must match when generating covariances.");
	}

	Eigen::MatrixXf &out = *C;
	for(int i = 0; i < A->rows(); i++){
		for(int j = 0; j < B->rows(); j++){
			out(i, j) = kernel->f(A->row(i), B->row(j), params);
		}
	}
}
