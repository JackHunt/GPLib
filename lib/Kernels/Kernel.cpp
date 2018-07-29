/*
BSD 3-Clause License

Copyright (c) 2018, Jack Miles Hunt
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

#include <Kernels/Kernel.hpp>

using namespace GPLib::Kernels;

template<typename T>
Kernel<T>::Kernel(const std::vector< std::string > &validParams, const ParameterSet<T> &params) :
    validParams(validParams), 
    params(params) {
    verifyParams();
}

template<typename T>
Kernel<T>::~Kernel() {
    //
}

template<typename T>
void Kernel<T>::verifyParams() const {
    // Check for missing parameters.
    for (const auto &p : validParams) {
		verifyParam(p);
    }

    // Check for invalid parameters.
    for (const auto p : params) {
        if (std::find(validParams.begin(), validParams.end(), p.first) == validParams.end()) {
            std::cout << "WARNING: Surplus parameter " + p.first + " being removed from parameter set!";
            params.erase(p.first);
        }
    }
}

template<typename T>
void Kernel<T>::verifyParam(const std::string &var) const {
	assert(params.find(var) != params.end());
}

template<typename T>
ParameterSet<T> Kernel<T>::getParameters() const {
	return params;
}

template<typename T>
void Kernel<T>::setParameters(const ParameterSet<T> &params) {
	this->params = params;
	verifyParams();
}