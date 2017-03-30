#ifndef GAUSSIAN_PROCESS_TYPEDEFS_HEADER
#define GAUSSIAN_PROCESS_TYPEDEFS_HEADER

#include <Eigen/Dense>

namespace GaussianProcess{
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
	typedef Eigen::VectorXd Vector;

}

#endif
