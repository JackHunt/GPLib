#ifndef LM_OPTIMISER_HEADER
#define LM_OPTIMISER_HEADER

#include "GPRegressor.h"
#include <memory>

namespace GaussianProcess{
	class LMOptimiser{
	private:
		//

	public:
		LMOptimiser(std::shared_ptr<GPRegressor> regressor);
		~LMOptimiser();

		ParamaterSet optimise(int iterations, float targetError);
	};
}

#endif
