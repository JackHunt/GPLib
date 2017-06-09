#include "GDOptimiser.h"

using namespace GaussianProcess;

GDOptimiser::GDOptimiser(std::shared_ptr<GPRegressor> regressor) :
	regressor(regressor) {
	//
}

GDOptimiser::~GDOptimiser() {
	//
}

ParamaterSet GDOptimiser::optimise(const ParamaterSet &params, int iterations,
								   double targetStepSize, double learnRate) {
	double stepNorm = 1e5;
	double newStep = 0.0;
	int iter = 0;
	
	//Get data matrix and ground truth.
	const Matrix &X = regressor->X_copy;
	const Vector &Y = regressor->Y_copy;

	//Get kernel and initial paramaters(copy).
	const std::shared_ptr<Kernel> &kernel = regressor->kernel;
	ParamaterSet optimParams(params);

	//Allocate storage.
	Matrix K(X.rows(), X.rows());
	Matrix K_deriv(K.rows(), K.cols());
	Matrix K_chol(K.rows(), K.cols());
	
	//Optimise for solution.
	while(stepNorm > targetStepSize && iter < iterations) {
		std::cout << "lambda: " << optimParams.at("lambda") << " sigma: " << optimParams.at("sigma") << std::endl;
		//Build covariance matrix.
		buildCovarianceMatrix<Matrix>(X, X, K, optimParams, kernel);
		K += Matrix::Identity(K.rows(), K.cols())*regressor->jitter;

		//Solve for alpha.
		jitterChol(K, K_chol);
		Vector alpha = K_chol.triangularView<Eigen::Lower>().solve(Y);
		
		//Build factor to reduce recomputing overhead.
		Matrix K_inv = K.inverse();
		Matrix factor = alpha * alpha.transpose() - K_inv;

		//Build covariance matrix jacobian and update, for each hyperparamater.
		stepNorm = 0.0;
		for(std::pair<const std::string, double> &par : optimParams) {
			const std::string &var = par.first;
			buildCovarianceMatrix<Matrix>(X, X, K_deriv, optimParams, kernel, var);
			newStep = -1.0*learnRate*(factor*K_deriv).trace();
			par.second += newStep;
			stepNorm += newStep*newStep;
		}
		stepNorm = sqrt(stepNorm);
		const double ll = logMarginalLikelihood(alpha, K, Y, K.rows());

		std::cout << "Iteration: " << iter << " Log-Likelihood: " << ll << std::endl;
		iter++;
	}
	return optimParams;
}

double GDOptimiser::logMarginalLikelihood(const Vector &alpha, const Matrix &K, const Vector &Y, int rows) {
	const double t1 = -0.5 * Y.transpose() * alpha;
	const double t2 = 0.5 * log(K.determinant());
	const double t3 = (static_cast<float>(rows) / 2.0) * log(2.0 * M_PI);
	return t1 - t2 - t3;
}
