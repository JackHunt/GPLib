#ifndef GP_REGRESSOR_HEADER
#define GP_REGRESSOR_HEADER

#include "Kernels.h"
#include "typedefs.h"

#include <Eigen/Dense>
#include <memory>

namespace GaussianProcess{
	class GPRegressor{
	private:		
		std::shared_ptr<Kernel> kernel;
		Matrix K, K_s, K_ss;
		Matrix L;
		Vector tmp, alpha;
		Vector f_s;
		Matrix v, v_s;
		Vector predDiff;
		Vector predSD;
		
		void buildCovarianceMatrix(const Eigen::Map<const Matrix> &A, const Eigen::Map<const Matrix> &B,
								   Matrix &C, const ParamaterSet &params);
		void jitterChol(const Matrix &A, Matrix &C);
		
	public:
		double runRegression(const double *trainData, const double *trainTruth, int trainRows, int trainCols,
							 const double *testData, const double *testTruth, int testRows, int testCols,
							 const ParamaterSet &params);
#ifdef WITH_PYTHON_BINDINGS
		double runRegression(const double *trainData, int trainCols, int trainRows, const double *trainTruth,
							 int trainTruthRows, const double *testData, int testCols, int testRows,
							 const double *testTruth, int testTruthRows, const ParamaterSet &params);
#endif
		const double *getMeans() const;
		const double *getCovariances() const;
		const double *getStdDev();
		
		GPRegressor(KernelType kernType = SQUARED_EXPONENTIAL);
		~GPRegressor();
	};
}

#endif
