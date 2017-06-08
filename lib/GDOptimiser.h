#ifndef LM_OPTIMISER_HEADER
#define LM_OPTIMISER_HEADER

#include "GPRegressor.h"
#include <memory>
#include <Eigen/Dense>

namespace GaussianProcess{
	class GDOptimiser{
	private:
		std::shared_ptr<GPRegressor> regressor;

	public:
		GDOptimiser(std::shared_ptr<GPRegressor> regressor);
		~GDOptimiser();

		ParamaterSet optimise(int iterations, double targetError);
	};
}

#endif
