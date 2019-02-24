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

#include <memory>

#include <Aliases.hpp>
#include <GaussianProcess.hpp>

#include <CPPUtils/Statistics/SampleStatistics.hpp>

namespace GPLib::Optimisation {
    template<typename>
    class OptimiserParameters {
    protected:
        const std::shared_ptr<GaussianProcess<T>> gp;
        const unsigned int maxEpochs;
        const T minConvergenceNorm;
        const unsigned int convergenceWindow;

    public:
        OptimiserParameters(std::shared_ptr<GaussianProcess<T>> gp,
                            unsigned int maxEpochs = 100,
                            T minConvergenceNorm = 1e-3,
                            unsigned int convergenceWindow = 5) {
            // Sanity check.
            assert(gp != nullptr);
            assert(minConvergenceNorm > 0);
        }

        virtual ~OptimiserParameters() {
            //
        }

        std::shared_ptr<GaussianProcess<T>> getGP() const {
            return gp;
        }

        unsigned int getMaxEpochs() const {
            return maxEpochs;
        }

        T getMinConvergenceNorm() const {
            return minConvergenceNorm;
        }

        unsigned int getConvergenceWindow() const {
            return convergenceWindow;
        }
    };

    template<typename T>
    class Optimiser {
    protected:
        const OptimiserParameters parameters;
        CPPUtils::Statistics::WindowedSampleStatistics<T, false> normMean;

    protected:
        Optimiser(const OptimiserParameters<T>& parameters) :
            parameters(parameters),
            normMean(parameters.getConvergenceWindow()) {
            //
        }

        bool converged(const Vector<T>& step) {
            normMean.provideSample(step.norm());
            return normMean.getEstimate() <= parameters.getMinConvergenceNorm();
        }

    public:
        virtual ~Optimiser() {
            //
        }

        virtual void operator()() = 0;
    };
}

#endif