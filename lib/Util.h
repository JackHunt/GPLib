#ifndef GAUSSIAN_PROCESS_UTIL_HEADER
#define GAUSSIAN_PROCESS_UTIL_HEADER

#include <memory>
#include <string>
#include "Kernels.h"

namespace GaussianProcess{
	inline void jitterChol(const Matrix &A, Matrix &C){
		const size_t rowsA = A.rows();
		const size_t colsA = A.cols();
		if(rowsA != colsA){
			throw std::runtime_error("Cannot take Cholesky Decomposition of non square matrix.");
		}
	
		Matrix jitter = Matrix::Identity(rowsA, colsA);
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

	template<typename T>
	inline void buildCovarianceMatrix(const T &A, const T &B, Matrix &C, const ParamaterSet &params,
									  const std::shared_ptr<Kernel> &kernel, const std::string &var = std::string("")){
		const size_t rowsA = A.rows();
		const size_t rowsB = B.rows();
#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic) collapse(2)
#endif
		for(size_t i = 0; i < rowsA; i++){
			for(size_t j = 0; j < rowsB; j++){
				if(var.compare("") != 0){
					C(i, j) = kernel->df(A.row(i), B.row(j), params, var);
				}else {
					C(i, j) = kernel->f(A.row(i), B.row(j), params);
				}
			}
		}
	}
}

#endif
