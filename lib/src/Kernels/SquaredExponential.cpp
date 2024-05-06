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

#include <GPLib/Kernels/SquaredExponential.hpp>

using namespace GPLib;

template<typename T>
SquaredExponential<T>::SquaredExponential() :
  Kernel<T>({ "sigma", "lambda" },
            { {"sigma", 1.0}, {"lambda", 1.0} }) {
  //
}

template<typename T>
SquaredExponential<T>::~SquaredExponential() {
  //
}

template<typename T>
T SquaredExponential<T>::f(const Vector<T>& a, const Vector<T>& b) const {
  assert(a.size() == b.size());

  const auto euc_dist_sq = (a - b).squaredNorm();

  const auto sigma = this->params.at("sigma");
  const auto lambda = this->params.at("lambda");

  return sigma * sigma * std::exp(-1 * (euc_dist_sq / (2 * lambda * lambda)));
}

template<typename T>
KernelGradient<T> SquaredExponential<T>::df(const Vector<T>& a, const Vector<T>& b,
                                            const std::optional<std::string>& grad_var) const {
  assert(a.size() == b.size());

  const auto euc_dist_sq = (a - b).squaredNorm();
  const auto sigma = this->params.at("sigma");
  const auto lambda = this->params.at("lambda");
  const auto lambda_sq = lambda * lambda;

  // dF/ dLambda
  const auto dLambda = [=, &a, &b]() -> T {
    return sigma * sigma * euc_dist_sq * std::exp((-0.5 * euc_dist_sq / lambda_sq));
  };

  // dF / dSigma
  const auto dSigma = [=, &a, &b] () -> T {
    return 2 * sigma * std::exp((-0.5 * euc_dist_sq) / lambda_sq);
  };

  if (!grad_var.has_value()) {
    Vector<T> nabla(2);
    nabla << dLambda(), dSigma();
    return nabla;
  }

  const auto& var = grad_var.value();
  this->verifyParam(var);

  if (var == "lambda") {
    return dLambda();
  }

  if (var == "sigma") {
    return dSigma();
  }

  // Unknown parameter.
  throw std::runtime_error("SquaredExponential: Parameter: \"" + var + "\" invalid.");
}

template<typename T>
Vector<T> SquaredExponential<T>::dfda(const Vector<T>& a, const Vector<T>& b) const {
  assert(a.size() == b.size());

  const T f_val = f(a, b);
  const T lambda = this->params.at("lambda");
  const auto df = (-1.0 / (lambda * lambda)) * f_val * (a - b);

  return df;
}

template<typename T>
Vector<T> SquaredExponential<T>::dfdb(const Vector<T>& a, const Vector<T>& b) const {
  assert(a.size() == b.size());

  return -1.0 * dfda(a, b);
}

namespace GPLib {
  template class SquaredExponential<float>;
  template class SquaredExponential<double>;
  template class SquaredExponential<long double>;
}
