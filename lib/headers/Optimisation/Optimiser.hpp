/*
BSD 3-Clause License

Copyright (c) 2019, Jack Miles Hunt
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

#ifndef GPLIB_OPTIMISER_HEADER
#define GPLIB_OPTIMISER_HEADER

#include <deque>
#include <numeric>

#include <Aliases.hpp>
#include <GaussianProcess.hpp>

namespace GPLib::Optimisation {
	template<typename T>
	class Optimiser {
	protected:
		const ParameterSet<T> parameters;
		const unsigned int maxEpochs;
		const T minConvergenceNorm;
		const unsigned int normSteps;
		
		std::deque<T> stepNorms;

	protected:
		Optimiser(const ParameterSet& parameters, 
			      unsigned int maxEpochs = 100,
			      T minConvergenceNorm = 1e-3,
			      unsigned int normSteps = 5) :
			parameters(parameters),
			maxEpochs(maxEpochs),
			minConvergenceNorm(minConvergenceNorm), 
		    normSteps(normSteps) {
			//
		}

		bool converged(const Vector<T>& step) {
			const auto norm = step.norm();
			
			if (stepNorms.size() < normSteps) {
				stepNorms.push_front(norm);
				return false;
			}

			stepNorms.pop_back();
			stepNorms.push_front(norm);
			
			auto meanNorm = std::accumulate(stepNorms.begin(), stepNorms.end(), 0);
			meanNorm /= static_cast<T>(normSteps);

			return meanNorm <= minConvergenceNorm;
		}

	public:
		virtual ~Optimiser() {
			//
		}

		virtual void operator()() = 0;
	};
}

#endif