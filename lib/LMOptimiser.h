#ifndef LM_OPTIMISER_HEADER
#define LM_OPTIMISER_HEADER

#include "GPRegressor.h"
#include <memory>
#include <map>
#include <string>

typedef std::map<std::string, float> ParamaterSet;

namespace GaussianProcess{
	class LMOPtimiser{
	private:
		//

	public:
		LMOptimiser(std::shared_ptr<GPRegressor> regressor);
		~LMOptimiser();

		ParamaterSet optimise(int iterations, float targetError);
	};
}

#endif
