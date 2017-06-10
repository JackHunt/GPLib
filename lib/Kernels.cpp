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

#include "Kernels.h"

using namespace GaussianProcess;

double SquaredExponential::f(const Vector &a, const Vector &b, const ParameterSet &params) const{
	if(params.find("sigma") == params.end() || params.find("lambda") == params.end()){
        throw std::runtime_error("Invalid Parameter set provided.");
	}

	if(a.size() != b.size()){
		throw std::runtime_error("Vector sizes must match in kernel.");
	}
	
	double sqEucDist = 0.0;
	const size_t size = a.size();
	for(size_t i = 0; i < size; i++){
		sqEucDist += (a(i) - b(i)) * (a(i) - b(i));
	}

    const double sigma = params.at("sigma");
	const double lambda = params.at("lambda");

	return sigma * sigma * expf(-1.0 * (sqEucDist / (2.0 * lambda * lambda)));
}

double SquaredExponential::df(const Vector &a, const Vector &b, const ParameterSet &params, const std::string &variable) const{
	if(params.find("sigma") == params.end() || params.find("lambda") == params.end()){
        throw std::runtime_error("Invalid Parameter set provided.");
	}

	if(a.size() != b.size()){
		throw std::runtime_error("Vector sizes must match in kernel.");
	}

	double sqEucDist = 0.0;
	const size_t size = a.size();
	for(size_t i = 0; i < size; i++){
		sqEucDist += (a(i) - b(i)) * (a(i) - b(i));
	}

	const double sigma = params.at("sigma");
	const double lambda = params.at("lambda");
	double deriv = 0.0;
	
	if(variable.compare("sigma") == 0) {
	    deriv = 2.0 * sigma * expf((-0.5 * sqEucDist) / lambda * lambda);
	}else if(variable.compare("lambda") == 0) {
		deriv = sigma * sigma * sqEucDist * expf((-0.5 * sqEucDist / lambda * lambda));
	}else {
		throw std::runtime_error("Invalid partial derivative requested.");
	}
	
	return deriv;
}
