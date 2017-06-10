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

#ifndef LM_OPTIMISER_HEADER
#define LM_OPTIMISER_HEADER

#include "GPRegressor.h"
#include "Util.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <Eigen/Dense>

namespace GaussianProcess{
	class GDOptimiser{
	private:
		double jitter = 1.0;
		Matrix K, K_deriv, K_chol;
		double logMarginalLikelihood(const Vector &alpha, const Matrix &K, const Vector &Y, int rows);

	public:
		GDOptimiser();
		~GDOptimiser();

//#ifdef WITH_PYTHON_BINDINGS
		ParamaterSet optimise(const double *trainData, int trainCols, int trainRows, const double *trainTruth,
							  int trainTruthRows, const ParamaterSet &params, const std::shared_ptr<Kernel> &kernel,
							  int iterations, double targetStepSize, double learnRate);
//#endif

		ParamaterSet optimise(const std::vector<double> &trainData, const std::vector<double> &trainTruth,
							  int trainRows, int trainCols, const ParamaterSet &params, const std::shared_ptr<Kernel> &kernel,
							  int iterations, double targetStepSize, double learnRate);

		
		ParamaterSet optimise(const double *trainData, const double *trainTruth, int trainRows, int trainCols,
							  const ParamaterSet &params, const std::shared_ptr<Kernel> &kernel, int iterations,
							  double targetStepSize, double learnRate);
		void setJitterFactor(double jitterFactor);
	};
}

#endif
