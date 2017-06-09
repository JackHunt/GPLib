#ifndef LM_OPTIMISER_HEADER
#define LM_OPTIMISER_HEADER

#include "GPRegressor.h"
#include "Util.h"
#include <memory>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>

namespace GaussianProcess{
	class GDOptimiser{
	private:
		std::shared_ptr<GPRegressor> regressor;
		double logLikelihood(const Vector &alpha, const Matrix &K, const Vector &Y, int rows);

	public:
		GDOptimiser(std::shared_ptr<GPRegressor> regressor);
		~GDOptimiser();

		ParamaterSet optimise(const ParamaterSet &params, int iterations, double targetLogLikelihood);
	};
}

#endif
