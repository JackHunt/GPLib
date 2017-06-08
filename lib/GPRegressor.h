#ifndef GP_REGRESSOR_HEADER
#define GP_REGRESSOR_HEADER

#include "Kernels.h"
#include "typedefs.h"

#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <iostream>

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
		double jitter = 1.0;
		
		void buildCovarianceMatrix(const Eigen::Map<const Matrix> &A, const Eigen::Map<const Matrix> &B,
								   Matrix &C, const ParamaterSet &params);
		void jitterChol(const Matrix &A, Matrix &C);
		
	public:
		double runRegression(const std::vector<double> &trainData, const std::vector<double> &trainTruth, int trainRows, int trainCols,
							 const std::vector<double> &testData, const std::vector<double> &testTruth, int testRows, int testCols,
							 const ParamaterSet &params);
		double runRegression(const double *trainData, const double *trainTruth, int trainRows, int trainCols,
							 const double *testData, const double *testTruth, int testRows, int testCols,
							 const ParamaterSet &params);
//#ifdef WITH_PYTHON_BINDINGS
		double runRegression(const double *trainData, int trainCols, int trainRows, const double *trainTruth,
							 int trainTruthRows, const double *testData, int testCols, int testRows,
							 const double *testTruth, int testTruthRows, const ParamaterSet &params);
//#endif
		std::vector<double> getMeans() const;
		std::vector<double> getCovariances() const;
		std::vector<double> getStdDev();
		void setJitterFactor(double jitterFactor);
		
		GPRegressor(KernelType kernType = SQUARED_EXPONENTIAL);
		~GPRegressor();
	};
}

#endif
