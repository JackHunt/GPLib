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

void GPRegressor::runRegression(const float *data, int rows, int cols){
	//
}

void GPRegressor::buildCovarianceMatrix(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B, Eigen::MatrixXf &C, const ParamaterSet &params){
	if(A.cols() != B.cols() || A.cols() != C.cols()){
		throw std::runtime_error("Matrix column dimensions must match when generating covariances.");
	}

	for(int i = 0; i < A.rows(); i++){
		for(int j = 0; j < B.rows(); j++){
			C(i, j) = kernel->f(A.row(i), B.row(j), params);
		}
	}
}
