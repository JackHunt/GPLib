#ifndef GAUSSIAN_PROCESS_UTIL_HEADER
#define GAUSSIAN_PROCESS_UTIL_HEADER

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
}

#endif
