#include "GDOptimiser.h"

using namespace GaussianProcess;

GDOptimiser::GDOptimiser(std::shared_ptr<GPRegressor> regressor) :
	regressor(regressor) {
	//
}

GDOptimiser::~GDOptimiser() {
	//
}

ParamaterSet GDOptimiser::optimise(const ParamaterSet &params, int iterations, double targetLogLikelihood) {
	double ll = 1e-5;
	int iter = 0;

	//Get covariance matrix and ground truth.
	const Matrix &K = regressor->K;
	const Matrix &X = regressor->X_copy;
	const Vector &Y = regressor->Y_copy;

    //Solve for alpha.
	Matrix K_chol(K.rows(), K.cols());
	jitterChol(K, K_chol);
	Vector alpha = K_chol.triangularView<Eigen::Lower>().solve(Y);

	//Build factor to reduce recomputing overhead.
	Matrix K_inv = K.inverse();
	Matrix factor = alpha * alpha.transpose() - K_inv;

	//Get kernel and initial paramaters(copy).
	const std::shared_ptr<Kernel> &kernel = regressor->kernel;
	ParamaterSet optimParams(params);
	
	//Optimise for solution.
	while(ll < targetLogLikelihood && iter < iterations) {
		for(std::pair<const std::string, double> &par : optimParams) {
			const std::string &var = par.first;
			double &varVal = par.second;

			double kernelGrad = 0.0;
			for(int i = 0; i < X.rows(); i++) {
				kernelGrad += kernel->df(X.row(i), X.row(i), params, var);
			}
			varVal -= (factor*kernelGrad).trace();
		}
		ll = logLikelihood(alpha, K, Y, K.rows());

		std::cout << "Iteration: " << iter << " Log-Likelihood: " << ll << std::endl;
		iter++;
	}
}

double GDOptimiser::logLikelihood(const Vector &alpha, const Matrix &K, const Vector &Y, int rows) {
	const double t1 = 0.5 * Y.transpose() * alpha;
	const double t2 = 0.5 * log(K.determinant());
	const double t3 = (static_cast<float>(rows) / 2.0) * log(2.0 * M_PI);
	return t1 - t2 - t3;
}
