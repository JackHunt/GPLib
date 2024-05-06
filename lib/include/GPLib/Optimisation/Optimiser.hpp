/*
BSD 3-Clause License

Copyright (c) 2020, Jack Miles Hunt
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

#include <CPPUtils/Statistics/SampleStatistics.hpp>

#include <GPLib/Aliases.hpp>
#include <GPLib/GaussianProcess.hpp>

namespace GPLib::Optimisation {
    template<typename T>
    inline Vector<T> paramsToVec(const ParameterSet<T>& params) {
        Vector<T> paramVec(params.size());
        for (const auto& p : params) {
            paramVec << p.second;
        }

        return paramVec;
    }

    template<typename T>
    inline ParameterSet<T> vecToParams(const ParameterSet<T>& paramSet,
                                       const Vector<T>& params) {
        // Verify consistency.
        const size_t numElems = params.rows() * params.cols();
        assert(paramSet.size() == numElems);

        // Copy params into new param set.
        size_t i = 0;
        ParameterSet<T> newParams(paramSet);
        for (auto& [k, v] : newParams) {
            v = params(i);
            i++;
        }

        return newParams;
    }

    template<typename T>
    class OptimiserParameters {
    protected:
        const std::shared_ptr<GaussianProcess<T>> gp;
        const MappedMatrix<T> X;
        const MappedVector<T> Y;

        const unsigned int maxIterations;
        const T minConvergenceNorm;
        const unsigned int convergenceWindow;

        OptimiserParameters(std::shared_ptr<GaussianProcess<T>> gp,
                            const MappedMatrix<T>& X,
                            const MappedVector<T>& Y,
                            unsigned int maxIterations = 100,
                            T minConvergenceNorm = 1e-3,
                            unsigned int convergenceWindow = 5) :
            gp(gp),
            X(X),
            Y(Y),
            maxIterations(maxIterations),
            minConvergenceNorm(minConvergenceNorm),
            convergenceWindow(convergenceWindow) {
            // Sanity check.
            assert(gp != nullptr);
            assert(minConvergenceNorm > 0);
        }

    public:
        virtual ~OptimiserParameters() {
            //
        }

        std::shared_ptr<GaussianProcess<T>> getGP() const {
            return gp;
        }

        const MappedMatrix<T>& getX() const {
            return X;
        }

        const MappedVector<T>& getY() const {
            return Y;
        }

        unsigned int getMaxIterations() const {
            return maxIterations;
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
        const OptimiserParameters<T> parameters;
        CPPUtils::Statistics::WindowedSampleStatistics<T, false> normMean;
        unsigned int iteration;

    protected:
        Optimiser(const OptimiserParameters<T>& parameters) :
            parameters(parameters),
            normMean(parameters.getConvergenceWindow()),
            iteration(0) {
            //
        }

        bool converged(const Vector<T>& step) {
            normMean.provideSample(step.norm());
            return std::get<T>(normMean.getEstimate()) <= parameters.getMinConvergenceNorm();
        }

    public:
        virtual ~Optimiser() {
            //
        }

        virtual void operator()() = 0;
    };
}

#endif
