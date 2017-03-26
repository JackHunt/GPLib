#include "Kernels.h"

using namespace GaussianProcess;

float SquaredExponential::f(const Eigen::VectorXf &a, const Eigen::VectorXf &b, const ParamaterSet &params) const{
	if(params.find("sigma") == params.end() || params.find("lambda") == params.end()){
		throw std::runtime_error("Invalid paramater set provided.");
	}

	if(a.size() != b.size()){
		throw std::runtime_error("Vector sizes must match in kernel.");
	}
	
	float sqEucDist = 0.0;
	for(int i = 0; i < a.size(); i++){
		sqEucDist += (a(i) - b(i)) * (a(i) - b(i));
	}

    const float sigma = params.at("sigma");
	const float lambda = params.at("lambda");

	return lambda * lambda * expf(-1.0 * (sqEucDist / (2.0 * sigma * sigma)));
}

float SquaredExponential::df(const Eigen::VectorXf &a, const Eigen::VectorXf &b, const std::string &var, const ParamaterSet &params) const{
	if(params.find("sigma") == params.end() || params.find("lambda") == params.end()){
		throw std::runtime_error("Invalid paramater set provided.");
	}

	if(a.size() != b.size()){
		throw std::runtime_error("Vector sizes must match in kernel.");
	}
}
