#ifndef GP_REGRESSOR_HEADER
#define GP_REGRESSOR_HEADER

#include "Kernels.h"
#include "typedefs.h"

#include <Eigen/Dense>

namespace GaussianProcess{
	class GPRegressor{
	private:		
		std::shared_ptr<Kernel> kernel;
		
		void buildCovarianceMatrix(const Eigen::Map<Matrix> &A, const Eigen::Map<Matrix> &B,
								   Matrix &C, const ParamaterSet &params);
		void jitterChol(const Eigen::Map<Matrix> &A, Matrix &C);
		
	public:
		void runRegression(const double *trainData, const double *trainTruth, int trainRows, int trainCols,
						   const double *testData, const double *testTruth, int testRows, int testCols,
						   const ParamaterSet &params);
		
		GPRegressor(KernelType kernType = SQUARED_EXPONENTIAL);
		~GPRegressor();
	};
}

#endif
