/*
  BSD 3-Clause License

  Copyright (c) 2024, Jack Miles Hunt
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
  inline Vector<T> params_to_vec(const ParameterSet<T>& params) {
    Vector<T> param_vec(params.size());
    for (const auto& p : params) {
      param_vec << p.second;
    }

    return param_vec;
  }

  template<typename T>
  inline ParameterSet<T> vec_to_params(const ParameterSet<T>& param_set,
                                       const Vector<T>& params) {
    // Verify consistency.
    assert(param_set.size() == params.rows() * params.cols());

    // Copy params into new param set.
    size_t i = 0;
    ParameterSet<T> new_params(param_set);
    for (auto& [k, v] : new_params) {
      v = params(i);
      i++;
    }

    return new_params;
  }

  template<typename T>
  class OptimiserParameters {
  protected:
    const std::shared_ptr<GaussianProcess<T>> gp;
    const MappedMatrix<T> X;
    const MappedVector<T> Y;

    const unsigned int max_iter;
    const T min_convergence_norm;
    const unsigned int convergence_window;

    OptimiserParameters(std::shared_ptr<GaussianProcess<T>> gp,
                        const MappedMatrix<T>& X,
                        const MappedVector<T>& Y,
                        unsigned int max_iter = 100,
                        T min_convergence_norm = 1e-3,
                        unsigned int convergence_window = 5) :
      gp(gp),
      X(X),
      Y(Y),
      max_iter(max_iter),
      min_convergence_norm(min_convergence_norm),
      convergence_window(convergence_window) {
      // Sanity check.
      assert(gp != nullptr);
      assert(min_convergence_norm > 0);
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

    unsigned int getMaxIter() const {
      return max_iter;
    }

    T getMinConvergenceNorm() const {
      return min_convergence_norm;
    }

    unsigned int getConvergenceWindow() const {
      return convergence_window;
    }
  };

  template<typename T>
  class Optimiser {
  protected:
    const OptimiserParameters<T> parameters;
    CPPUtils::Statistics::WindowedSampleStatistics<T, false> norm_mean;
    unsigned int iteration;

  protected:
    Optimiser(const OptimiserParameters<T>& parameters) :
      parameters(parameters),
      norm_mean(parameters.getConvergenceWindow()),
      iteration(0) {
      //
    }

    bool converged(const Vector<T>& step) {
      norm_mean.provideSample(step.norm());
      return std::get<T>(norm_mean.getEstimate()) <= parameters.getMinConvergenceNorm();
    }

  public:
    virtual ~Optimiser() {
      //
    }

    virtual void operator()() = 0;
  };
}

#endif
