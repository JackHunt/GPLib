#include "Kernels.h"

using namespace GaussianProcess;

double SquaredExponential::f(const Vector &a, const Vector &b, const ParamaterSet &params) const{
	if(params.find("sigma") == params.end() || params.find("lambda") == params.end()){
		throw std::runtime_error("Invalid paramater set provided.");
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

	return lambda * lambda * expf(-1.0 * (sqEucDist / (2.0 * sigma * sigma)));
}

double SquaredExponential::df(const Vector &a, const Vector &b, const std::string &var, const ParamaterSet &params) const{
	if(params.find("sigma") == params.end() || params.find("lambda") == params.end()){
		throw std::runtime_error("Invalid paramater set provided.");
	}

	if(a.size() != b.size()){
		throw std::runtime_error("Vector sizes must match in kernel.");
	}
}
