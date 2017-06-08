#include "GDOptimiser.h"

using namespace GaussianProcess;

GDOptimiser::GDOptimiser(std::shared_ptr<GPRegressor> regressor) :
	regressor(regressor) {
	//
}

GDOptimiser::~GDOptimiser() {
	//
}

ParamaterSet GDOptimiser::optimise(int iterations, double targetError) {
	double error = 0.0;
	int iter = 0;

	const Matrix &K = regressor->K;
//	const Eigen::Map<const Vector> &Y = regressor->Y;
	
	Vector alpha;
	Vector nabla;
	
	while(error > targetError && iter < iterations) {
		iter++;
	}
}
