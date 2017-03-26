#ifndef GP_REGRESSOR_HEADER
#define GP_REGRESSOR_HEADER

#include "Kernels.h"
#include <Eigen/Dense>

namespace GaussianProcess{
	class GPRegressor{
	private:
		std::shared_ptr<Kernel> kernel;
		void buildCovarianceMatrix(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B, Eigen::MatrixXf &C, const ParamaterSet &params);
		
	public:
		void runRegression(const float *data, int rows, int cols);
		
		GPRegressor(KernelType kernType = SQUARED_EXPONENTIAL);
		~GPRegressor();
	};
}

#endif
