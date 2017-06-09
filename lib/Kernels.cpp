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

	return sigma * sigma * expf(-1.0 * (sqEucDist / (2.0 * lambda * lambda)));
}

double SquaredExponential::df(const Vector &a, const Vector &b, const ParamaterSet &params, const std::string &variable) const{
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
