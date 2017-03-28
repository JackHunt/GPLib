#ifndef GP_REGRESSOR_HEADER
#define GP_REGRESSOR_HEADER

#include "Kernels.h"
#include <Eigen/Dense>

namespace GaussianProcess{
	class GPRegressor{
	private:
		std::shared_ptr<Kernel> kernel;
		std::shared_ptr<Eigen::Map<Eigen::MatrixXf> > X, X_s;
		std::shared_ptr<Eigen::Map<Eigen::VectorXf> > Y, Y_s;
		std::shared_ptr<Eigen::MatrixXf> K, K_s, K_ss;
		
		void computeCovarianceMatrices(const ParamaterSet &params);
		void buildCovarianceMatrix(const std::shared_ptr<Eigen::Map<Eigen::MatrixXf> > &A,
								   const std::shared_ptr<Eigen::Map<Eigen::MatrixXf> > &B,
								   std::shared_ptr<Eigen::MatrixXf> &C, const ParamaterSet &params);
		
	public:
		void runRegression(const float *trainData, const float *trainTruth, int trainRows, int trainCols,
						   const float *testData, const float testTruth, int testRows, int testCols,
						   const ParamaterSet &params);
		
		GPRegressor(KernelType kernType = SQUARED_EXPONENTIAL);
		~GPRegressor();
	};
}

#endif
