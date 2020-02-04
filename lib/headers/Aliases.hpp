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

#ifndef GPLIB_ALIASES_HEADER
#define GPLIB_ALIASES_HEADER

#include <string>
#include <map>
#include <variant>
#include <tuple>

#include <Eigen/Dense>

namespace GPLib::Types {
    // Generic Row-Major Matrix.
    template<typename T>
    using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Generic Row-Major Matrix wrapper for C style buffers.
    template<typename T>
    using MappedMatrix = Eigen::Map<const Matrix<T>>;

    // Generic Column-Vector.
    template<typename T>
    using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    // Generic Column-Vector wrapper for C style buffers.
    template<typename T>
    using MappedVector = Eigen::Map<Vector<T>>;

    // Variable name, value - for kernel parameters.
    template<typename T>
    using ParameterSet = std::map<std::string, T>;

    // Matrix and Covariance pair.
    template<typename T>
    using MeanCov = std::tuple<Matrix<T>, Matrix<T>>;

    // Matrix, Covariance and Error pair.
    template<typename T>
    using MeanCovErr = std::tuple<Matrix<T>, Matrix<T>, T>;

    // GP return type. Either MeanCov or MeanCovErr.
    template<typename T>
    using GPOutput = std::variant<MeanCov<T>, MeanCovErr<T>>;

    // Kernel hyperparameter gradient. Scalar for single variable, vector for all.
    template<typename T>
    using KernelGradient = std::variant<T, Vector<T>>;
}

using namespace GPLib::Types;

#endif
