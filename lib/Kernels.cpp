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

Vector SquaredExponential::df(const Vector &a, const Vector &b, const ParamaterSet &params) const{
	if(params.find("sigma") == params.end() || params.find("lambda") == params.end()){
		throw std::runtime_error("Invalid paramater set provided.");
	}

	if(a.size() != b.size()){
		throw std::runtime_error("Vector sizes must match in kernel.");
	}

	const double sigma = params.at("sigma");
	const double lambda = params.at("lambda");
	double sqEucDist = 0.0;
	const size_t size = a.size();
	for(size_t i = 0; i < size; i++){
		sqEucDist += (a(i) - b(i)) * (a(i) - b(i));
	}
	
	double dFdS = lambda * lambda * sqEucDist * expf((-0.5 * sqEucDist / sigma * sigma));
	double dFdL = 2.0 * lambda * expf((-0.5 * sqEucDist) / sigma * sigma);

	Vector nabla;
	nabla << dFdS, dFdL;
	return nabla;
}
