/*
BSD 3-Clause License

Copyright (c) 2017, Jack Miles Hunt
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
