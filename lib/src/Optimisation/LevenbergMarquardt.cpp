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

#include <GPLib/Optimisation/LevenbergMarquardt.hpp>

using namespace GPLib;
using namespace GPLib::Optimisation;

template<typename T>
LevenbergMarquardt<T>::LevenbergMarquardt(const LMParameters<T>& parameters) :
  Optimiser<T>(parameters),
  lambda(parameters.getLambda()) {
  // Allocate hessian and gradient.
  auto gp = parameters.getGP();
  gradK.resize(gp->getK().rows(), gp->getK().cols());
}

template<typename T>
LevenbergMarquardt<T>::~LevenbergMarquardt() {
  //
}

template<typename T>
void LevenbergMarquardt<T>::operator()() {
  auto gp = this->parameters.getGP();
  auto alpha = gp->getAlpha();
  auto K = gp->getK();
  auto X = gp->getX();

  const auto nParams = gp->getKernel()->getParameters().size();
  const auto I = Matrix<T>::Identity(nParams, nParams);
  Vector<T> nabla(nParams);

  size_t iteration = 0;
  while (iteration < this->parameters.getMaxIterations()) {
    // Compute log-likelihood of the GP.
    const auto logLik = logLikelihood<T>(alpha, K, this->parameters.getY());

    // Compute gradient w.r.t K.
    const auto dfdk = alpha * alpha.transpose() - K.inverse();

    // Compute gradients of K.
    for (const auto& [var, val] : gp->getKernel()->getParameters()) {
      buildCovarianceMatrix<T>(X, X.transpose(), gradK, gp->getKernel(), var);
      nabla << (dfdk * gradK).trace();
    }

    // Compute Hessian.
    const auto H = nabla.transpose() * nabla;

    // Compute step.
    const auto cholH = Eigen::LLT<Matrix<T>>(H + lambda * I).matrixL();
    const auto step = cholH.solve(nabla);

    // Compute new params.
    const ParameterSet<T> kernelParams(gp->getKernel()->getParameters());
    const auto updated = paramsToVec(kernelParams) - step;
    const auto newKernelParams = vecToParams<T>(kernelParams, updated);

    // Re-estimate the GP.
    gp->getKernel()->setParameters(newKernelParams);
    gp->compute(gp->getX(), gp->getY());

    const auto logLikelihoodNew = logLikelihood<T>(alpha, K, this->parameters.getY());
    if (logLikelihoodNew >= logLik) {
      lambda *= 10;
    }
    else {
      lambda /= 10;
      gp->getKernel()->setParameters(kernelParams);
    }

    if (this->converged(step)) {
      break;
    }
  }
}

namespace GPLib::Optimisation {
  template class LevenbergMarquardt<float>;
  template class LevenbergMarquardt<double>;
  template class LevenbergMarquardt<long double>;
}
